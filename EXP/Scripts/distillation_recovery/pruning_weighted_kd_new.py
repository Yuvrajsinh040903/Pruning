import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import gc
import time
import pandas as pd
from pathlib import Path
from torch.cuda import empty_cache
#from config import DATA_TEMPLATE, LANGS, SRC_COLUMN, TGT_COLUMN, TGT_LANGS
from pruning_core import (
    LANG_NAMES, MAX_ROWS, STRATEGIES, Calibration, build_multilingual_inputs, drop_layer,
    load_model, load_tokenizer, select_layer,
)
from recovery_kd_new import RecoveryConfig, build_recovery_data, heal_pruned_model_kd, make_target
SRC_LANG = "eng"
SRC_COLUMN = "source"
TGT_COLUMN = "target"
TGT_LANGS = ["hin", "tel", "guj", "tam", "ben", "urd", "kan"]

LANGS = {
    "eng": {"flores": "eng_Latn"},
    "hin": {"flores": "hin_Deva"},
    "tel": {"flores": "tel_Telu"},
    "guj": {"flores": "guj_Gujr"},
    "tam": {"flores": "tam_Taml"},
    "ben": {"flores": "ben_Beng"},
    "urd": {"flores": "urd_Arab"},
    "kan": {"flores": "kan_Knda"},
}

DATA_TEMPLATE = "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/flores_data/{split}/eng-{tgt}.tsv"

model_name = "sarvamai/sarvam-translate"
calib_per_lang = 10
bpcc_dir = "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/bpcc_data/train"
num_iterations = 8
strategy_name = "cka_cos"
strategy = STRATEGIES[strategy_name]

# set to a list of original layer ids to skip selection and remove exactly these, e.g. [23, 9, 3, 10, 6, 11, 21, 19]
forced_removals = None

variants = {
    "mse": dict(mse_weight=1.0),
    "mse_kl": dict(mse_weight=1.0, kl_weight=1.0),
    "mse_kl_ce": dict(mse_weight=1.0, kl_weight=1.0, ce_weight=0.5),
}
variant = "mse_kl_ce"
recovery_cfg = RecoveryConfig(**variants[variant], train_per_lang=4000, eval_per_lang=50, lr=1e-4)

lang_names = {tgt: LANG_NAMES[LANGS[tgt]["flores"]] for tgt in TGT_LANGS}
calib_sources = [(DATA_TEMPLATE.format(split="dev", tgt=tgt), lang_names[tgt]) for tgt in TGT_LANGS]
recovery_sources = [(f"{bpcc_dir}/eng-{tgt}.tsv", lang_names[tgt]) for tgt in TGT_LANGS]

order_tag = "forced" if forced_removals else strategy_name
cur_time = time.asctime().replace(" ", "_").replace(":", "-")
save_root = Path("/home/extra_storage_dipti/yuvrajsinh.bodana/Pruning_Experiments")
p = save_root / f"EXP_{order_tag}_kd_base_{variant}_{cur_time}_{model_name}"
p.mkdir(parents=True, exist_ok=True)

tokenizer = load_tokenizer(model_name)
model = load_model(model_name)
teacher = load_model(model_name).eval().requires_grad_(False)
calib_inputs = build_multilingual_inputs(tokenizer, calib_sources, SRC_COLUMN, calib_per_lang)
calib = Calibration(calib_inputs, model.device, MAX_ROWS)
train_ds, eval_ds = build_recovery_data(tokenizer, recovery_sources, SRC_COLUMN, TGT_COLUMN, recovery_cfg)
print(f"Calibration prompts: {calib_inputs['input_ids'].shape[0]} | recovery train/eval: {len(train_ds)}/{len(eval_ds)}")

active_ids = list(range(len(model.model.layers)))
initial_layer_count = len(active_ids)
removed = []

for iteration in range(num_iterations):
    print(f"\n===== ITERATION {iteration + 1}/{num_iterations} =====")
    if forced_removals:
        best = active_ids.index(forced_removals[iteration])
    else:
        best, table = select_layer(model, calib, active_ids, strategy)
        table.insert(0, "iteration", iteration + 1)
        table.to_csv(p / f"prune_iteration_{iteration + 1}.tsv", sep="\t", index=False)
    layer_id = active_ids.pop(best)
    removed.append(layer_id)

    drop_layer(model, best)
    target = make_target(active_ids, best, initial_layer_count)
    print(f"Removed layer {layer_id} | training layer {target.trainable_orig} towards base layer {target.base_idx} output")
    model = heal_pruned_model_kd(model, teacher, tokenizer, iteration + 1, p, target, train_ds, eval_ds, recovery_cfg)
    gc.collect()
    empty_cache()

pd.DataFrame({"iteration": range(1, len(removed) + 1), "removed_layer_id": removed}).to_csv(
    p / "removed_layers.tsv", sep="\t", index=False
)
print(f"\nRemoved (original ids): {removed}")
print(f"Remaining (original ids): {active_ids}")

final_layer_count = len(model.model.layers)
save_dir = p / f"Pruned_Model_{initial_layer_count}_to_{final_layer_count}_{order_tag}_{variant}"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Saved to {save_dir}")
