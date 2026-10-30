import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.cuda import empty_cache
import gc
from LoRA_healing import heal_pruned_model
import time
from pathlib import Path
from huggingface_hub import login

login(token="")

cur_time = time.asctime().replace(" ","_").replace(":","-")
p = Path(f"/home/priyanka/Pruning/EXP/Experiments/EXP_CKA_{cur_time}_wo_rec")
p.mkdir(exist_ok=True)

# ==========================================
# CKA FUNCTIONS (Provided)
# ==========================================
def gram_linear(x):
    return x.dot(x.T)

def gram_rbf(x, threshold=1.0):
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def center_gram(gram, unbiased=False):
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram

def cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    # print(f"The values are: {normalization_x} and {normalization_y}")
    return scaled_hsic / (normalization_x * normalization_y)

def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

def feature_space_linear_cka(features_x, features_y, debiased=False):
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)

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
# 2. PRE-COMPUTE BASE MODEL STATES (CKA SETUP)
# ==========================================
print("\nLoading Base Model to compute reference hidden states...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype=torch.float16, attn_implementation="eager"
)
inputs = inputs.to(base_model.device)

with torch.inference_mode():
    base_outputs = base_model(**inputs, output_hidden_states=True)

# Extract last token states and convert to float32 numpy array for stable CKA computation
base_last_token_states = base_outputs.hidden_states[-1][:, -1, :] 
base_np = base_last_token_states.to(torch.float32).cpu().numpy()

# Pre-compute the Gram matrix for the base model to save time inside the loop
gram_base = gram_linear(base_np)

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
    for index, layer in enumerate(model_to_update.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = index
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
    model_to_update.config.num_hidden_layers = len(model_to_update.model.layers)

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
    
    best_cka_score = -1.0
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
        pruned_np = pruned_last_token_states.to(torch.float32).cpu().numpy()
        
        # 4c. Compute CKA
        gram_pruned = gram_linear(pruned_np)
        current_cka = cka(gram_base, gram_pruned)
        
        iteration_results.append({
            "iteration": iteration + 1,
            "original_layer_id": absolute_layer_id,
            "cka_score": current_cka
        })
        
        print(f"Tested dropping original layer {absolute_layer_id:02d} | CKA Score: {current_cka:.4f}")
        
        if current_cka > best_cka_score:
            best_cka_score = current_cka
            best_relative_idx = i
            
        # 4d. Restore the layer for the next test
        model.model.layers = nn.ModuleList(original_layers)
        reindex_layers(model)
        empty_cache()
    
    # 4e. Execute Permanent Removal
    best_absolute_id = active_layer_ids[best_relative_idx]
    print(f"\n[!] Iteration {iteration + 1} Decision: Permanently removing original layer {best_absolute_id} (CKA Score: {best_cka_score:.4f})")
    
    original_layers.pop(best_relative_idx)
    model.model.layers = nn.ModuleList(original_layers)
    reindex_layers(model)
    
    active_layer_ids.pop(best_relative_idx)
    removed_history.append(best_absolute_id)
    
    # 4f. Save results
    df = pd.DataFrame(iteration_results)
    df.to_csv(f"prune_iteration_{iteration + 1}.tsv", sep='\t', index=False)

    # model = heal_pruned_model(model, tokenizer, iteration + 1, p)

print("\n========== PRUNING COMPLETE ==========")
print(f"Total layers removed: {num_iterations}")
print(f"Order of original layers removed: {removed_history}")
print(f"Remaining active original layers: {active_layer_ids}")

# ==========================================
# 5. SAVE FINAL PRUNED MODEL
# ==========================================
final_layer_count = len(model.model.layers)
save_dir =p/f"Pruned_Model_{initial_layer_count}_to_{final_layer_count}_CKA"

print(f"\nSaving the final pruned model to: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model and tokenizer successfully saved!")