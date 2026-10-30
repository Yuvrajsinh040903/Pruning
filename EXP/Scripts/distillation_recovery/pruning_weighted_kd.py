import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import gc
import time
import pandas as pd
from pathlib import Path
from torch.cuda import empty_cache
from pruning_core import (
    MAX_ROWS, STRATEGIES, Calibration, build_inputs, drop_layer,
    load_model, load_tokenizer, select_layer,
)
from recovery_kd import RecoveryConfig, build_recovery_data, heal_pruned_model_kd, make_gap

model_name = "meta-llama/Llama-3.2-3B-Instruct"
dataset_path = "/home/yuvrajsinh.bodana/Pruning/EXP/Data/Flores_Dev.tsv"
num_samples = 100
num_iterations = 4
strategy_name = "cos_cka"
strategy = {**STRATEGIES[strategy_name], "weight": 0.5}

recovery_path = "/home/yuvrajsinh.bodana/Pruning/EXP/Data/Recovery.tsv"
src_col, tgt_col = "src", "tgt"
recovery_cfg = RecoveryConfig(use_kl=False)

variant = "mse_kl" if recovery_cfg.use_kl else "mse"
cur_time = time.asctime().replace(" ", "_").replace(":", "-")
save_root = Path("/home/extra_storage_dipti/yuvrajsinh.bodana/Pruning_Experiments")
p = save_root / f"EXP_{strategy_name}_w{strategy['weight']}_kd_{variant}_{cur_time}"
p.mkdir(parents=True, exist_ok=True)

tokenizer = load_tokenizer(model_name)
model = load_model(model_name)
calib = Calibration(build_inputs(tokenizer, dataset_path, num_samples), model.device, MAX_ROWS)
train_ds, eval_ds = build_recovery_data(tokenizer, recovery_path, src_col, tgt_col, recovery_cfg)

active_ids = list(range(len(model.model.layers)))
initial_layer_count = len(active_ids)
removed = []

for iteration in range(num_iterations):
    print(f"\n===== ITERATION {iteration + 1}/{num_iterations} =====")
    best, table = select_layer(model, calib, active_ids, strategy)
    layer_id = active_ids.pop(best)
    removed.append(layer_id)
    print(f"Removed original layer {layer_id}")

    table.insert(0, "iteration", iteration + 1)
    table.to_csv(p / f"prune_iteration_{iteration + 1}.tsv", sep="\t", index=False)

    removed_layer = drop_layer(model, best)
    gap = make_gap(model, removed_layer, best)
    model = heal_pruned_model_kd(model, tokenizer, iteration + 1, p, gap, train_ds, eval_ds, recovery_cfg)
    del gap, removed_layer
    gc.collect()
    empty_cache()

pd.DataFrame({"iteration": range(1, len(removed) + 1), "removed_layer_id": removed}).to_csv(
    p / "removed_layers.tsv", sep="\t", index=False
)
print(f"\nRemoved (original ids): {removed}")
print(f"Remaining (original ids): {active_ids}")

final_layer_count = len(model.model.layers)
save_dir = p / f"Pruned_Model_{initial_layer_count}_to_{final_layer_count}_{strategy_name}_{variant}"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Saved to {save_dir}")