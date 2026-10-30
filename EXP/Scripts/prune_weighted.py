import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import gc
import time
import pandas as pd
from pathlib import Path
from torch.cuda import empty_cache
from LoRA_healing import heal_pruned_model
from pruning_core import (
    MAX_ROWS, STRATEGIES, Calibration, build_inputs, load_model,
    load_tokenizer, prune_step,
)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
dataset_path = "/home/yuvrajsinh.bodana/Pruning/EXP/Data/Flores_Dev.tsv"
num_samples = 100
num_iterations = 8
strategy_name = "cos_cka"
strategy = {**STRATEGIES[strategy_name], "weight": 0.5}

cur_time = time.asctime().replace(" ", "_").replace(":", "-")
save_root = Path("/home/extra_storage_dipti/yuvrajsinh.bodana/Pruning_Experiments")
p = save_root / f"EXP_{strategy_name}_w{strategy['weight']}_{cur_time}"
p.mkdir(parents=True, exist_ok=True)

tokenizer = load_tokenizer(model_name)
model = load_model(model_name)
calib = Calibration(build_inputs(tokenizer, dataset_path, num_samples), model.device, MAX_ROWS)

active_ids = list(range(len(model.model.layers)))
initial_layer_count = len(active_ids)
removed = []

for iteration in range(num_iterations):
    print(f"\n===== ITERATION {iteration + 1}/{num_iterations} =====")
    layer_id, table = prune_step(model, calib, active_ids, strategy)
    removed.append(layer_id)
    print(f"Removed original layer {layer_id}")

    table.insert(0, "iteration", iteration + 1)
    table.to_csv(p / f"prune_iteration_{iteration + 1}.tsv", sep="\t", index=False)

    model = heal_pruned_model(model, tokenizer, iteration + 1, p)
    gc.collect()
    empty_cache()

pd.DataFrame({"iteration": range(1, len(removed) + 1), "removed_layer_id": removed}).to_csv(
    p / "removed_layers.tsv", sep="\t", index=False
)
print(f"\nRemoved (original ids): {removed}")
print(f"Remaining (original ids): {active_ids}")

final_layer_count = len(model.model.layers)
save_dir = p / f"Pruned_Model_{initial_layer_count}_to_{final_layer_count}_{strategy_name}"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Saved to {save_dir}")