import csv
import gc
import json
import os
from collections import Counter

import torch

import config as C
from data import load_pairs
from evaluation import (allocated_mb, coverage, fidelity_after, fidelity_before, model_device, score,
                        translate)
from models import load
from pruning import prune_model, save_pruned, weight_stats
from selection import build_keep_sets, count_tokens


def cached(path, compute):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    value = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    return value


def to_counter(d):
    return Counter({int(k): v for k, v in d.items()})


def cache_file(model_name, name):
    return os.path.join(C.OUTPUT_DIR, "cache", model_name, name)


def get_counts(model_name, codec, tgt):
    def compute():
        src_lines, tgt_lines = load_pairs(C.SELECT_SPLIT, tgt, C.MAX_SELECT_LINES)
        src_c, tgt_c = count_tokens(codec, src_lines, tgt_lines, C.SRC_LANG, tgt)
        return {"src": src_c, "tgt": tgt_c}
    d = cached(cache_file(model_name, f"counts_{tgt}.json"), compute)
    return to_counter(d["src"]), to_counter(d["tgt"])


def get_generated_counts(model_name, model, codec, tgt):
    def compute():
        src_lines, _ = load_pairs(C.SELECT_SPLIT, tgt, C.MODEL_OUTPUT_LINES)
        _, ids, _ = translate(model, codec, src_lines, C.SRC_LANG, tgt, C.GEN_KWARGS[model_name], C.GEN_BATCH_SIZE)
        return Counter(i for seq in ids for i in seq)
    return to_counter(cached(cache_file(model_name, f"generated_{tgt}.json"), compute))


def get_baseline(model_name, model, codec, tgt, sources, refs):
    def compute():
        hyps, _, perf = translate(model, codec, sources, C.SRC_LANG, tgt, C.GEN_KWARGS[model_name], C.GEN_BATCH_SIZE)
        return {"hyps": hyps, **score(hyps, refs), **perf}
    return cached(cache_file(model_name, f"baseline_{tgt}.json"), compute)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiment(model_name, setting, mode, group, tgts, out_dir):
    model, codec = load(model_name, C.MODEL_PATHS[model_name], C.LANGS, C.DTYPE, C.DEVICE)
    device = model_device(model)
    vocab_orig = model.config.vocab_size
    base_weights, base_alloc = weight_stats(model), allocated_mb(device)

    evals = {t: load_pairs(C.EVAL_SPLIT, t, C.MAX_EVAL_LINES) for t in tgts}
    counts = {t: get_counts(model_name, codec, t) for t in tgts}
    generated = {t: get_generated_counts(model_name, model, codec, t) for t in tgts} if setting["model_outputs"] else {}
    baselines = {t: get_baseline(model_name, model, codec, t, *evals[t]) for t in tgts}

    keep_src, keep_tgt, per_lang = build_keep_sets(codec, setting, C.SRC_LANG, tgts, counts, generated, mode)
    cov = {t: coverage(codec, *evals[t], C.SRC_LANG, t, keep_src, keep_tgt) for t in tgts}
    fid_state = {t: fidelity_before(model, codec, *evals[t], C.SRC_LANG, t, keep_src, keep_tgt,
                                    C.FIDELITY_SENTENCES, C.TF_BATCH_SIZE) for t in tgts}

    meta = prune_model(model, keep_src, keep_tgt, mode)
    pruned_codec = codec.pruned(keep_src, keep_tgt)
    free_memory()
    pruned_weights, pruned_alloc = weight_stats(model), allocated_mb(device)

    rows = []
    for t in tgts:
        fid = fidelity_after(model, pruned_codec, C.SRC_LANG, t, fid_state[t], C.TF_BATCH_SIZE)
        sources, refs = evals[t]
        hyps, _, perf = translate(model, pruned_codec, sources, C.SRC_LANG, t, C.GEN_KWARGS[model_name], C.GEN_BATCH_SIZE)
        base = baselines[t]
        quality = score(hyps, refs)
        rows.append({
            "model": model_name, "setting": setting["name"], "mode": mode, "group": group, "tgt": t,
            "vocab_orig": vocab_orig,"vocab_src": meta["vocab_src"], "vocab_tgt": meta["vocab_tgt"],
            "vocab_lang_tgt": per_lang[t],
            "params_M_base": base_weights["params_M"], "params_M_pruned": pruned_weights["params_M"],
            "embed_params_M_base": base_weights["embed_params_M"], "embed_params_M_pruned": pruned_weights["embed_params_M"],
            "weights_MB_base": base_weights["weights_MB"], "weights_MB_pruned": pruned_weights["weights_MB"],
            "alloc_MB_base": base_alloc, "alloc_MB_pruned": pruned_alloc,
            "peak_MB_base": base["peak_MB"], "peak_MB_pruned": perf["peak_MB"],
            "seconds_base": base["seconds"], "seconds_pruned": perf["seconds"],
            "bleu_base": base["bleu"], "bleu_pruned": quality["bleu"],
            "chrf++_base": base["chrf++"], "chrf++_pruned": quality["chrf++"],
            "identical_outputs": sum(h == b for h, b in zip(hyps, base["hyps"])) / len(hyps),
            **cov[t], **fid,
        })
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"hyps.{t}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(hyps) + "\n")

    if C.SAVE_MODELS:
        save_pruned(model, pruned_codec, {**meta, "setting": setting, "targets": tgts,
                                          "keep_src": keep_src, "keep_tgt": keep_tgt}, out_dir)
    del model, fid_state
    free_memory()
    return rows


def print_rows(rows):
    for r in rows:
        print(f"  {r['model']:<12} {r['setting']:<9} {r['mode']:<6} {r['group']:<4} {r['tgt']}  "
              f"V {r['vocab_orig']}->{r['vocab_src']}/{r['vocab_tgt']}  "
              f"params {r['params_M_base']:.0f}->{r['params_M_pruned']:.0f}M  "
              f"weights {r['weights_MB_base']:.0f}->{r['weights_MB_pruned']:.0f}MB  "
              f"BLEU {r['bleu_base']:.2f}->{r['bleu_pruned']:.2f}  chrF++ {r['chrf++_base']:.2f}->{r['chrf++_pruned']:.2f}  "
              f"same {100 * r['identical_outputs']:.1f}%  src_oov {100 * r['src_oov_rate']:.2f}%  "
              f"p_removed {r['removed_mass_mean']:.2e}  dlogit {r['logit_max_abs_diff']:.1e}")


def write_csv(rows, path):
    if not rows:
        return
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.manual_seed(C.SEED)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    all_rows = []
    for model_name in C.MODELS:
        for setting in C.SELECTION_SETTINGS:
            for mode in C.EMBED_MODES:
                for group, tgts in C.TARGET_GROUPS.items():
                    out_dir = os.path.join(C.OUTPUT_DIR, "models", model_name, f"{setting['name']}_{mode}_{group}")
                    result_file = os.path.join(out_dir, "result.json")
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            rows = json.load(f)
                    else:
                        print(f"\n== {model_name} | {setting['name']} | {mode} | {group}")
                        rows = run_experiment(model_name, setting, mode, group, tgts, out_dir)
                        with open(result_file, "w") as f:
                            json.dump(rows, f, indent=2)
                    print_rows(rows)
                    all_rows.extend(rows)
                    write_csv(all_rows, os.path.join(C.OUTPUT_DIR, "results.csv"))


if __name__ == "__main__":
    main()
