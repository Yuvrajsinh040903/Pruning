import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.cuda import empty_cache
from LoRA_healing import heal_pruned_model
import gc
import time
from pathlib import Path
from huggingface_hub import login

login(token="")

cur_time = time.asctime().replace(" ","_").replace(":","-")
p = Path(f"/home/priyanka/Pruning/EXP/Experiments/EXP_Cosine_{cur_time}_wo_rec")
p.mkdir(exist_ok=True)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
model_name = "meta-llama/Llama-3.2-3B-Instruct"
dataset_name = "/home/priyanka/Pruning/EXP/Data/Flores_Dev.tsv"
num_iterations = 4  # Number of layers to remove permanently

print("Loading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

ds = load_dataset("csv", data_files=dataset_name, delimiter='\t', split='train')
ds = ds.select(range(10))
# ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
sentences_en = ds['src']

def make_prompts(text):
    prompt = f"""Translate the following sentence from English to Hindi:
{text}
RESPONSE FORMAT: <ONLY TRANSLATED TEXT>"""
    return [{"role": "user", "content": prompt}]

messages = [make_prompts(i) for i in sentences_en]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True
)

# ==========================================
# 2. PRE-COMPUTE BASE MODEL STATES
# ==========================================
print("\nLoading Base Model to compute reference hidden states...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
)
inputs = inputs.to(base_model.device)

with torch.inference_mode():
    # Forward pass (prefill only), no generation
    base_outputs = base_model(**inputs, output_hidden_states=True)

# We take the very last layer [-1], and the very last token [-1] for all batches [:]
base_last_token_states = base_outputs.hidden_states[-1][:, -1, :] 

# Free the base model completely to save VRAM
del base_model
empty_cache()
gc.collect()
print("Base reference states saved. Base model removed from memory.")

# ==========================================
# 3. LOAD WORKING MODEL & HELPER FUNC
# ==========================================
print("\nLoading Working Model for iterative pruning...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype=torch.float16, attn_implementation="eager"
)

def reindex_layers(model_to_update):
    """Updates the internal layer indices to prevent attention routing errors."""
    for index, layer in enumerate(model_to_update.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = index
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
    model_to_update.config.num_hidden_layers = len(model_to_update.model.layers)

# Keep track of absolute original layer IDs and counts
active_layer_ids = list(range(len(model.model.layers)))
initial_layer_count = len(active_layer_ids)
removed_history = []

# ==========================================
# 4. ITERATIVE PRUNING LOOP
# ==========================================
for iteration in range(num_iterations):
    print(f"\n========== ITERATION {iteration + 1}/{num_iterations} ==========")
    
    original_layers = list(model.model.layers)
    num_current_layers = len(original_layers)
    
    best_similarity = -1.0
    best_relative_idx = -1
    
    iteration_results = []
    
    for i in range(num_current_layers):
        absolute_layer_id = active_layer_ids[i]
        
        # 4a. Temporarily drop the layer
        temp_layers = original_layers[:i] + original_layers[i+1:]
        model.model.layers = nn.ModuleList(temp_layers)
        reindex_layers(model)
        
        # 4b. Forward pass on pruned network
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            
        pruned_last_token_states = outputs.hidden_states[-1][:, -1, :]
        
        # 4c. Compute Cosine Similarity
        sim = F.cosine_similarity(base_last_token_states, pruned_last_token_states, dim=-1)
        avg_sim = sim.mean().item()
        
        iteration_results.append({
            "iteration": iteration + 1,
            "original_layer_id": absolute_layer_id,
            "avg_cosine_similarity": avg_sim
        })
        
        print(f"Tested dropping original layer {absolute_layer_id:02d} | Avg Cosine Similarity: {avg_sim:.4f}")
        
        if avg_sim > best_similarity:
            best_similarity = avg_sim
            best_relative_idx = i
            
        # 4d. Restore the layer for the next test
        model.model.layers = nn.ModuleList(original_layers)
        reindex_layers(model)
        empty_cache()
    
    # 4e. Execute Permanent Removal
    best_absolute_id = active_layer_ids[best_relative_idx]
    print(f"\n[!] Iteration {iteration + 1} Decision: Permanently removing original layer {best_absolute_id}")
    
    original_layers.pop(best_relative_idx)
    model.model.layers = nn.ModuleList(original_layers)
    reindex_layers(model)
    
    active_layer_ids.pop(best_relative_idx)
    removed_history.append(best_absolute_id)
    
    # 4f. Save results
    df = pd.DataFrame(iteration_results)
    df.to_csv(f"prune_iteration_{iteration + 1}.tsv", sep='\t', index=False)
    
    # ==========================================
    # 4g. THE HEALING PHASE
    # ==========================================
    # Pass the pruned model to the healing function. 
    # It will train, merge, save, and return a healthy model for the next loop.
    # model = heal_pruned_model(model, tokenizer, iteration + 1, p)

print("\n========== PRUNING COMPLETE ==========")
print(f"Total layers removed: {num_iterations}")
print(f"Order of original layers removed: {removed_history}")
print(f"Remaining active original layers: {active_layer_ids}")

# ==========================================
# 5. SAVE FINAL PRUNED MODEL
# ==========================================
final_layer_count = len(model.model.layers)
save_dir = p/f"Pruned_Model_{initial_layer_count}_to_{final_layer_count}_cosine"

print(f"\nSaving the final pruned model to: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model and tokenizer successfully saved!")