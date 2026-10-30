import os
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, OFTConfig,get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
import gc
from pathlib import Path

_CACHED_PARALLEL_DS = None


# def get_prepared_dataset():
#     global _CACHED_PARALLEL_DS
#     if _CACHED_PARALLEL_DS is not None:
#         return _CACHED_PARALLEL_DS

#     print("Preparing IN22-Gen dataset for healing phase...")
#     dataset_name = "ai4bharat/IN22-Gen"
#     ds = load_dataset(dataset_name, split="test")

#     lang_maps = {
#         "Hindi": "hin_Deva",
#         "Gujarati": "guj_Gujr",
#         "Tamil": "tam_Taml",
#         "Telugu": "tel_Telu",
#         "Bengali": "ben_Beng",
#         "English": "eng_Latn"
#     }
    
#     source_langs = ["English"]
#     # target_langs = ["Gujarati", "Hindi", "Telugu", "Tamil", "Bengali"]
#     target_langs = ["Hindi"]
    

#     def make_prompts_batched(batch):
#         all_messages = []
#         batch_size = len(batch[list(batch.keys())[0]])
        
#         for i in range(batch_size):
#             for src in source_langs:
#                 for tgt in target_langs:
#                     src_sent = batch[lang_maps[src]][i]
#                     tgt_sent = batch[lang_maps[tgt]][i]
#                     prompt = f"""Translate following sentence from {src} to {tgt}:\n{src_sent}"""
#                     all_messages.append([
#                         {"role": "user", "content": prompt},
#                         {"role": "assistant", "content": tgt_sent}
#                     ])
#         return {"messages": all_messages}

#     parallel_ds = ds.map(
#         make_prompts_batched, 
#         batched=True, 
#         remove_columns=ds.column_names
#     )
    
#     _CACHED_PARALLEL_DS = parallel_ds.shuffle(seed=42)
#     return _CACHED_PARALLEL_DS

# def get_prepared_dataset():
#     global _CACHED_PARALLEL_DS
#     if _CACHED_PARALLEL_DS is not None:
#         return _CACHED_PARALLEL_DS

#     print("Preparing Flores+ devtest (En->Hi) dataset for healing phase...")
    
#     ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
#     ds_hi = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split="devtest")
    
#     combined_ds = Dataset.from_dict({
#         "english": ds_en["text"],
#         "hindi": ds_hi["text"]
#     })
    
#     def make_prompts_batched(batch):
#         all_messages = []
#         for src_sent, tgt_sent in zip(batch["english"], batch["hindi"]):
#             prompt = f"Translate the following sentence from English to Hindi:\n{src_sent}\nRESPONSE FORMAT: <ONLY TRANSLATED TEXT>"
            
#             all_messages.append([
#                 {"role": "user", "content": prompt},
#                 {"role": "assistant", "content": tgt_sent}
#             ])
            
#         return {"messages": all_messages}

#     _CACHED_PARALLEL_DS = combined_ds.map(
#         make_prompts_batched, 
#         batched=True, 
#         remove_columns=["english", "hindi"]
#     )
    
#     _CACHED_PARALLEL_DS = _CACHED_PARALLEL_DS.shuffle(seed=42)
#     return _CACHED_PARALLEL_DS

def get_prepared_dataset(dev_size=500, seed=42):
    """
    Loads BPCC-15K and splits into train/dev.
    dev_size: number of examples held out for evaluation/early stopping.
    """
    print("Preparing BPCC-15K (En->Hi) dataset for healing phase...")

    ds = load_dataset("csv", data_files="/home/priyanka/Pruning/EXP/Data/BPCC_seed_latest.tsv", delimiter='\t', split='train')
    ds = ds.select(range(15000))

    def make_prompts_batched(batch):
        all_messages = []
        for src_sent, tgt_sent in zip(batch["src"], batch["tgt"]):
            prompt = f"Translate the following sentence from English to Hindi:\n{src_sent}\nRESPONSE FORMAT: <ONLY TRANSLATED TEXT>"
            all_messages.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": tgt_sent}
            ])
        return {"messages": all_messages}

    ds_final = ds.map(
        make_prompts_batched,
        batched=True,
        remove_columns=ds.column_names
    )

    # Shuffle before splitting so dev set isn't just the tail of the file
    ds_final = ds_final.shuffle(seed=seed)

    split = ds_final.train_test_split(test_size=dev_size, seed=seed)
    train_ds = split["train"]
    dev_ds = split["test"]

    print(f"Train examples: {len(train_ds)} | Dev examples: {len(dev_ds)}")
    return train_ds, dev_ds


def heal_pruned_model_oft(model, tokenizer, iteration, p):
    """
    Fine-tunes the pruned model using QLoRA/LoRA to recover performance,
    merges the weights, saves it, and returns the merged model.
    """
    print(f"\n[+] Starting Healing Phase for Iteration {iteration}")
    save_path = Path(p / "pruning_recovery")
    save_path.mkdir(exist_ok=True)
    save_dir = f"{save_path}/cosine_recovery_{iteration}_bpcc"

    train_ds, dev_ds = get_prepared_dataset(dev_size=500)

    oft_config = OFTConfig(
    task_type="CAUSAL_LM",
    peft_type="OFT",
    r=32,
    oft_block_size=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],    
)

    sft_config = SFTConfig(
        output_dir=f"{save_dir}_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=1,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="messages",
        loss_type="nll",

        # --- Early stopping config ---
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,          # must match eval_steps for load_best_model_at_end
        save_total_limit=2,      # keep only best + latest checkpoint, save disk
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=oft_config,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training adapters...")
    trainer.train()

    print(f"Merging LoRA weights back into base model...")
    merged_model = trainer.model.merge_and_unload()

    merged_model = merged_model.to(torch.bfloat16)

    print(f"Saving healed model to {save_dir}")
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("[+] Healing Phase Complete.\n")
    return merged_model