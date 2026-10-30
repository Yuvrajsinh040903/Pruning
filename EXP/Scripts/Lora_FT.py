import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback,AutoModelForCausalLM, AutoTokenizer
import gc
from pathlib import Path

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


def heal_pruned_model(model, tokenizer, p):
    """
    Fine-tunes the pruned model using QLoRA/LoRA to recover performance,
    merges the weights, saves it, and returns the merged model.
    """
    p = Path(p)
    save_path = Path(p / "pruning_recovery")
    save_path.mkdir(exist_ok=True)
    save_dir = f"{save_path}/final_recovery_bpcc"

    train_ds, dev_ds = get_prepared_dataset(dev_size=500)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        use_rslora=True,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    sft_config = SFTConfig(
        output_dir=f"{save_dir}_checkpoints",
        num_train_epochs=5,
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
        peft_config=lora_config,
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

if __name__ == "__main__":
    model_path = "/home/priyanka/Pruning/EXP/Experiments/EXP_CKA_Wed_Sep_16_00-30-28_2026_wo_rec/Pruned_Model_28_to_24_CKA"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    heal_pruned_model(model, tokenizer, p=model_path)