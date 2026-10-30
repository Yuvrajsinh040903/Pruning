import os
# os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
# os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import time
import torch.nn as nn
import pandas as pd
from pathlib import Path
from pruning_core import (
    MAX_ROWS, STRATEGIES, Calibration, build_inputs, load_model,
    load_tokenizer, prune_step, reindex_layers,
)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
dataset_path = "/home/priyanka/Pruning/EXP/Data/Flores_Dev.tsv"
num_samples = 100
num_remove = 4
metric_cols = ["g_cosine", "g_cka", "l_cosine", "l_cka"]

cur_time = time.asctime().replace(" ", "_").replace(":", "-")
save_root = Path("/home/priyanka/Pruning/EXP/Pruning_Experiments")
out_dir = save_root / f"PILOT_{cur_time}"
out_dir.mkdir(parents=True, exist_ok=True)

tokenizer = load_tokenizer(model_name)
model = load_model(model_name)
calib = Calibration(build_inputs(tokenizer, dataset_path, num_samples), model.device, MAX_ROWS)
full_layers = list(model.model.layers)


def run_strategy(name, strategy):
    model.model.layers = nn.ModuleList(full_layers)
    reindex_layers(model)
    active_ids = list(range(len(full_layers)))
    removed = []
    for it in range(num_remove):
        layer_id, table = prune_step(model, calib, active_ids, strategy)
        table.insert(0, "iteration", it + 1)
        table.to_csv(out_dir / f"{name}_iter{it + 1}.tsv", sep="\t", index=False)
        removed.append(layer_id)
        print(f"[{name}] iter {it + 1}: removed layer {layer_id}")
        if it == 0:
            first_table = table
    oneshot = first_table.nlargest(num_remove, "score")["layer_id"].tolist()
    return {"removed": removed, "oneshot": oneshot, "first_table": first_table}


def jaccard_matrix(sets):
    names = list(sets)
    return pd.DataFrame(
        [[len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for b in names] for a in names],
        index=names, columns=names,
    )


results = {name: run_strategy(name, s) for name, s in STRATEGIES.items()}

summary = pd.DataFrame([
    {
        "strategy": name,
        "iterative_order": r["removed"],
        "iterative_set": sorted(r["removed"]),
        "oneshot_top": sorted(r["oneshot"]),
    }
    for name, r in results.items()
])
summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False)

overlap_iter = jaccard_matrix({n: set(r["removed"]) for n, r in results.items()})
overlap_oneshot = jaccard_matrix({n: set(r["oneshot"]) for n, r in results.items()})
overlap_iter.to_csv(out_dir / "overlap_iterative.tsv", sep="\t")
overlap_oneshot.to_csv(out_dir / "overlap_oneshot.tsv", sep="\t")

base_table = next(iter(results.values()))["first_table"]
corr = base_table[metric_cols].corr(method="spearman")
corr.to_csv(out_dir / "metric_rank_correlation.tsv", sep="\t")

print("\n" + summary.to_string(index=False))
print("\nJaccard overlap (iterative removal sets)\n", overlap_iter.round(2))
print("\nJaccard overlap (one-shot top-k sets)\n", overlap_oneshot.round(2))
print("\nSpearman correlation between metrics (iteration 1)\n", corr.round(2))
print(f"\nSaved to {out_dir}")