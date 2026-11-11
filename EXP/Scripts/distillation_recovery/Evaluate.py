import argparse
import csv
import gc
import resource
from pathlib import Path
 
import pandas as pd
 
from metrics import NEURAL_METRICS, build_metric
 
# metric name -> column name in _summary.tsv (matches what the inference script writes)
SUMMARY_COL = {"bleu": "BLEU", "chrf++": "chrF++", "comet": "COMET",
               "comtail-da": "COMTAIL-DA", "comtail-qe": "COMTAIL-QE"}
SUMMARY_ORDER = ["pair", "n", "chrF++", "BLEU", "COMET", "COMTAIL-DA", "COMTAIL-QE"]
 
# Column names: the inference script writes source/target, older files use src/tgt
COLUMN_ALIASES = {"src": ["source", "src"], "ref": ["target", "tgt"], "mt": ["mt"]}
 
# The inference script writes with QUOTE_NONE + escapechar "\\", so read back the same way
TSV_READ = dict(sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\",
                keep_default_na=False, dtype=str, encoding="utf-8")
TSV_WRITE = dict(sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
 
 
def log_mem(tag):
    """Current and peak RAM of this process (what Slurm's --mem limit counts), plus GPU memory."""
    cur = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2   # KB on Linux -> GB
    msg = f"[mem] {tag}: RAM now {cur:.1f} GB, peak {peak:.1f} GB" if cur else f"[mem] {tag}: RAM peak {peak:.1f} GB"
    try:
        import torch
        if torch.cuda.is_available():
            msg += f" | GPU alloc {torch.cuda.memory_allocated() / 1024**3:.1f} GB"
    except ImportError:
        pass
    print(msg, flush=True)
 
 
def pick_column(df, key, path):
    for c in COLUMN_ALIASES[key]:
        if c in df.columns:
            return c
    raise ValueError(f"{path}: no {key} column (tried {COLUMN_ALIASES[key]}), found {list(df.columns)}")
 
 
def pair_files(folder):
    return sorted(f for f in Path(folder).glob("*-*.tsv") if not f.name.startswith("_"))
 
 
def score_file(path, metric, overwrite):
    """Run one metric on one file. Returns (n_rows, corpus_score or None if skipped)."""
    df = pd.read_csv(path, **TSV_READ)
    col = f"{metric.name}_score"
    if col in df.columns and not overwrite:
        print(f"|>   {path.name}: {metric.name} already present, skipping", flush=True)
        return len(df), None
    srcs = df[pick_column(df, "src", path)].tolist()
    refs = df[pick_column(df, "ref", path)].tolist()
    mts = df[pick_column(df, "mt", path)].tolist()
    print(f"|>   {path.name}: running {metric.name} on {len(df)} sentences", flush=True)
    sent_scores, corpus_score = metric.compute(srcs, mts, refs)
    df[col] = sent_scores
    df.to_csv(path, **TSV_WRITE)
    return len(df), corpus_score
 
 
def update_summary(folder, rows):
    """Merge new scores into <folder>/_summary.tsv, keeping existing columns and pairs."""
    summ_file = folder / "_summary.tsv"
    new = pd.DataFrame(list(rows.values())).set_index("pair")
    if summ_file.exists():
        old = pd.read_csv(summ_file, sep="\t").set_index("pair")
        summ = old.combine_first(new)          # keep every pair from both
        summ.update(new)                       # new scores overwrite old ones
    else:
        summ = new
    summ = summ.reset_index().sort_values("pair")
    order = [c for c in SUMMARY_ORDER if c in summ.columns]
    summ = summ[order + [c for c in summ.columns if c not in order]]
    summ.to_csv(summ_file, sep="\t", index=False)
    print(f"|> {summ_file}\n{summ.to_string(index=False)}\n")
 
 
def rebuild_summary_all(inference_dir):
    """Rebuild <inference>/_summary_all.tsv if it exists (one row per test set and pair)."""
    out = inference_dir / "_summary_all.tsv"
    if not out.exists():
        return
    frames = []
    for f in sorted(inference_dir.glob("*/_summary.tsv")):
        s = pd.read_csv(f, sep="\t")
        s.insert(0, "test_set", f.parent.name)
        frames.append(s)
    if frames:
        pd.concat(frames).to_csv(out, sep="\t", index=False)
        print(f"|> rebuilt {out}")
 
 
def free(metric):
    del metric
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
 
 
def main(args):
    folders = [Path(f) for f in args.folders]
    for f in folders:
        if not pair_files(f):
            raise FileNotFoundError(f"no eng-xxx.tsv files in {f}")
 
    comet_kwargs = {"model_name": args.comet_model, "batch_size": args.batch_size,
                    "num_workers": args.num_workers}
    comtail_kwargs = {
        name: {"comtail_dir": args.comtail_dir, "ckpt_path": ckpt,
               "batch_size": args.batch_size, "num_workers": args.num_workers}
        for name, ckpt in [("comtail-da", args.comtail_da_ckpt), ("comtail-qe", args.comtail_qe_ckpt)]
    }
 
    # rows[folder][pair] = {"pair":..., "n":..., "BLEU":..., ...}
    rows = {f: {} for f in folders}
    log_mem("start")
 
    for name in args.metrics:
        # Skip loading a model if every file already has this metric
        col = f"{name}_score"
        pending = [p for f in folders for p in pair_files(f)
                   if args.overwrite or col not in pd.read_csv(p, nrows=0, **{k: v for k, v in TSV_READ.items()}).columns]
        if not pending:
            print(f"|> {name}: already present in every file, skipping")
            continue
 
        print(f"\n|> ===== {name}: {len(pending)} files =====", flush=True)
        metric = build_metric(name, comet_kwargs, comtail_kwargs)
        log_mem(f"{name} loaded")
 
        for folder in folders:
            for path in pair_files(folder):
                n, score = score_file(path, metric, args.overwrite)
                row = rows[folder].setdefault(path.stem, {"pair": path.stem, "n": n})
                if score is not None:
                    row[SUMMARY_COL[name]] = round(score, 4 if name in NEURAL_METRICS else 2)
            log_mem(f"{name} done on {folder.name}")
 
        free(metric)
        log_mem(f"{name} freed")
 
        # Write summaries after each metric, so a crash later doesn't lose this one
        for folder in folders:
            if rows[folder]:
                update_summary(folder, rows[folder])
                rebuild_summary_all(folder.parent)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True,
                        help="one or more folders holding eng-xxx.tsv files and _summary.tsv")
    parser.add_argument("--metrics", nargs="+", default=["comet"],
                        help="subset of: bleu chrf++ comet comtail-da comtail-qe (default: comet)")
    parser.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comtail_dir", default="comtail-models",
                        help="folder holding COMTAIL-DA/ and COMTAIL-QE/ checkpoint dirs")
    parser.add_argument("--comtail_da_ckpt", default=None,
                        help="explicit COMTAIL-DA .ckpt path (overrides --comtail_dir)")
    parser.add_argument("--comtail_qe_ckpt", default=None,
                        help="explicit COMTAIL-QE .ckpt path (overrides --comtail_dir)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers for COMET/COMTAIL; 0 = no extra processes (least RAM)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute metrics whose sentence-score column already exists")
    main(parser.parse_args())

