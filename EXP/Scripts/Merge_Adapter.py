import torch
from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer
from pathlib import Path
import sys

model_path = Path(sys.argv[1])
print("Model Mentioned is : ", model_path)

# Load tokenizer first
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load PEFT model directly (handles base model + adapter automatically)
print("Loading PEFT model with adapters...")
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Better for Qwen3 + vLLM
    device_map="auto",
    trust_remote_code=True
)

# Merge adapters into base model
print("Merging adapters with base model...")
merged_model = model.merge_and_unload()
merged_model.resize_token_embeddings(len(tokenizer))  # Fix embeddings if needed

# Save merged model + tokenizer
output_dir = model_path.parent / f"merged-{model_path.name}"
print(f"Saving merged model to: {output_dir}")

merged_model.save_pretrained(
    output_dir, 
    safe_serialization=True  # Safetensors format for vLLM
)
tokenizer.save_pretrained(output_dir)

print(f"✅ Merged model saved to: {output_dir}")
print("Ready for vLLM inference!")
