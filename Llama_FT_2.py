import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import SFTTrainer, SFTConfig

from huggingface_hub import login

login(token="")

# =========================================================
# CONFIG
# =========================================================

model_name = "meta-llama/Llama-3.2-1B-Instruct"

output_dir = "/scratch/yuvraj.bodana/Llama_1B_inst_in22_flores_ft"

train_file = "./train_dataset.tsv"
valid_file = "./valid_dataset.tsv"

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# LOAD DATASETS
# =========================================================

train_dataset = load_dataset(
    "csv",
    data_files=train_file,
    delimiter="\t",
    split="train"
)

valid_dataset = load_dataset(
    "csv",
    data_files=valid_file,
    delimiter="\t",
    split="train"
)

# =========================================================
# REMOVE UNWANTED COLUMN
# =========================================================

if "Unnamed: 0" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns("Unnamed: 0")

if "Unnamed: 0" in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns("Unnamed: 0")

# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_function(example):

    user = example["prompt"].split("Target:")[0].strip()
    assistant = example["prompt"].split("Target:")[1].strip()

    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }

train_dataset = train_dataset.map(
    preprocess_function,
    remove_columns=["prompt"]
)

valid_dataset = valid_dataset.map(
    preprocess_function,
    remove_columns=["prompt"]
)

train_dataset = train_dataset.shuffle(seed=42)
valid_dataset = valid_dataset.shuffle(seed=42)

# =========================================================
# QLORA CONFIG
# =========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",

    # RTX 2080Ti -> use float16 instead of bfloat16
    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=False,
)

# =========================================================
# MODEL
# =========================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

# print("The mode is on : ", model.device)
model.config.use_cache = False

# =========================================================
# LORA CONFIG
# =========================================================

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj"
    ],

    init_lora_weights="gaussian",

    modules_to_save=["lm_head"],
)

# =========================================================
# SFT CONFIG
# =========================================================

sft_config = SFTConfig(

    output_dir=output_dir,

    # Training
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,

    # Optimization
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",

    # Logging
    logging_steps=10,
    report_to="none",

    # Saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=100,

    # IMPORTANT FOR EARLY STOPPING
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Precision
    bf16=False,
    fp16=False,

    # TRL
    # assistant_only_loss=True,
    completion_only_loss=True,
    dataset_text_field="messages",

    # Optional
    max_length=2048,
)

# =========================================================
# TRAINER
# =========================================================

trainer = SFTTrainer(
    model=model,

    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    processing_class=tokenizer,

    peft_config=peft_config,

    args=sft_config,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0
        )
    ],
)

# =========================================================
# TRAIN
# =========================================================

trainer.train()

# =========================================================
# SAVE BEST MODEL
# =========================================================

trainer.save_model(output_dir)