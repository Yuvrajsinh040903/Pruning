from transformers import AutoModelForCausalLM
import torch
from huggingface_hub import login
login(token="")
model_name = "sarvamai/sarvam-translate"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16)

print(model)
print("Download Completed..!!")
