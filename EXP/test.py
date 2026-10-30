import os
os.environ['HF_HOME'] = "/ssd_scratch/yuvraj.bodana/huggingface"
os.environ['HF_DATASETS_CACHE'] = "/ssd_scratch/yuvraj.bodana/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import pandas as pd
from torch.cuda import empty_cache
import gc
from datasets import load_dataset

ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn")


model_1_name = "meta-llama/Llama-3.2-3B-Instruct"
model_2_name = "meta-llama/Llama-3.2-1B-Instruct"

model_1 = AutoModelForCausalLM.from_pretrained(model_2_name, device_map="auto", dtype=torch.float16, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_2_name)

model_2 = AutoModelForCausalLM.from_pretrained(model_2_name, device_map="auto", dtype=torch.float16, attn_implementation="eager")
tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)

tokenizer_2.pad_token = tokenizer.eos_token
tokenizer_2.padding_side = "left"

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

sentences_en = list(ds['devtest']['text'])[:10]

save_data = {
    "source":sentences_en
}

def make_prompts(text):
    prompt = f"""Translate the following sentence from English to Hindi:
{text}
RESPONSE FORMAT: <ONLY TRANSLATED TEXT>"""

    messages = [
        {"role":"user", "content":prompt}
    ]

    return messages

messages = []
for i in sentences_en:
    messages.append(make_prompts(i))

inputs = tokenizer_2.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
    padding=True
).to(model_2.device)

with torch.inference_mode():
    outputs = model_2.generate(**inputs, max_new_tokens=128, output_hidden_states=True, return_dict_in_generate=True)

generated_ids = outputs.sequences

print("Original Model:\n")
original_output = []
for index,i in enumerate(generated_ids):
    # print(sentences_en[index])
    op = tokenizer_2.decode(i[inputs["input_ids"][index].shape[-1]:], skip_special_tokens=True)
    original_output.append(op)
    # print(op)
    # print("---")

save_data["base_model_output"] = original_output

# Pruning


for rm_layer in range(16):
    model_1 = AutoModelForCausalLM.from_pretrained(model_2_name, device_map="auto", dtype=torch.float16, attn_implementation="eager")
    original_layers = list(model_1.model.layers) 
    layers_to_drop = [rm_layer] 
    
    new_layers = [layer for i, layer in enumerate(original_layers) if i not in layers_to_drop]
    
    model_1.model.layers = nn.ModuleList(new_layers)
    
    model_1.config.num_hidden_layers = len(new_layers)
    
    for index, layer in enumerate(model_1.model.layers):
        # Update the index in the attention module (where the error originates)
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = index
        
        # Update the index on the layer itself (good practice for compatibility)
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
    
    inputs_2 = tokenizer_2.apply_chat_template(
    	messages,
    	add_generation_prompt=True,
    	tokenize=True,
    	return_dict=True,
    	return_tensors="pt",
        padding=True
    ).to(model_1.device)
    
    with torch.inference_mode():
        outputs_2 = model_1.generate(**inputs_2, max_new_tokens=128, output_hidden_states=True, return_dict_in_generate=True)
    
    generated_ids_2 = outputs_2.sequences
    
    print("Pruned Model:\n")
    pruned_output = []
    for index,i in enumerate(generated_ids_2):
        # print(sentences_en[index])
        op = tokenizer_2.decode(i[inputs["input_ids"][index].shape[-1]:], skip_special_tokens=True)
        pruned_output.append(op)
        # print(op)
        # print("---")
    
    save_data["pruned_model_output"] = pruned_output
    
    df = pd.DataFrame.from_dict(save_data)
    df.to_csv(f"prune_layer_{layers_to_drop[0]}.tsv", sep='\t')

    del model_1
    empty_cache()
    gc.collect()
    print(f"Completed layer {rm_layer}...")