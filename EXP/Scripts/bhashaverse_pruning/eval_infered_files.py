import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import gc
import glob

import torch
from sacrebleu.metrics import BLEU, CHRF

# Folders written by inference.py; every <src>_<tgt>.tsv inside gets per-sentence score columns
FOLDERS = [
    "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/distilled/flores_extra_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/distilled/flores_pruning_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/distilled/in22_conv_extra_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/distilled/in22_conv_pruning_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/distilled/in22_gen_extra_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/distilled/in22_gen_pruning_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/teacher/flores_extra_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/teacher/flores_pruning_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/teacher/in22_conv_extra_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/teacher/in22_conv_pruning_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/teacher/in22_gen_extra_inference",
"/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/inference/teacher/in22_gen_pruning_inference"

]
METRICS = ["bleu", "chrf++", "comet"]
RECOMPUTE = False
SUMMARY_NAME = "_summary.tsv"

BLEU_TOKENIZE = "13a"
# Hub name, or a local path to a .ckpt file
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 32
COMET_GPUS = 1 if torch.cuda.is_available() else 0


def clean(text):
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f if line.strip()]
    return header, rows


def write_tsv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(clean(str(row.get(h, ""))) for h in header) + "\n")
    os.replace(tmp, path)


def pair_files(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*.tsv")) if os.path.basename(p) != SUMMARY_NAME)


def score_bleu(rows):
    sent = BLEU(tokenize=BLEU_TOKENIZE, effective_order=True)
    scores = [sent.sentence_score(r["hypothesis"], [r["target"]]).score for r in rows]
    corpus = BLEU(tokenize=BLEU_TOKENIZE).corpus_score([r["hypothesis"] for r in rows], [[r["target"] for r in rows]])
    return scores, corpus.score


def score_chrf(rows):
    chrf = CHRF(word_order=2)
    scores = [chrf.sentence_score(r["hypothesis"], [r["target"]]).score for r in rows]
    corpus = chrf.corpus_score([r["hypothesis"] for r in rows], [[r["target"] for r in rows]])
    return scores, corpus.score


class Comet:
    def __init__(self):
        from comet import download_model, load_from_checkpoint
        path = COMET_MODEL if COMET_MODEL.endswith(".ckpt") else download_model(COMET_MODEL)
        self.model = load_from_checkpoint(path)

    def __call__(self, rows):
        data = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["target"]} for r in rows]
        # num_workers=1 keeps CPU RAM low; extra data-loader processes copy the model
        out = self.model.predict(data, batch_size=COMET_BATCH_SIZE, gpus=COMET_GPUS, num_workers=1, progress_bar=False)
        scores = [100 * s for s in out.scores]
        return scores, sum(scores) / len(scores)


def make_scorer(metric):
    return {"bleu": lambda: score_bleu, "chrf++": lambda: score_chrf, "comet": Comet}[metric]()


def main():
    files = [p for folder in FOLDERS for p in pair_files(folder)]
    print(f"{len(files)} files in {len(FOLDERS)} folders")
    # summary[folder][pair][metric] = system score; one metric at a time so only one scorer is in memory
    summary = {}
    for metric in METRICS:
        scorer = None
        for path in files:
            header, rows = read_tsv(path)
            folder, pair = os.path.dirname(path), os.path.basename(path)[:-4]
            if not rows:
                continue
            if metric in header and not RECOMPUTE and all(r.get(metric, "") != "" for r in rows):
                values = [float(r[metric]) for r in rows]
                # Recompute corpus BLEU/chrF++ from the text; sentence scores do not average to them
                system = (score_bleu(rows)[1] if metric == "bleu" else score_chrf(rows)[1] if metric == "chrf++"
                          else sum(values) / len(values))
            else:
                scorer = scorer or make_scorer(metric)
                values, system = scorer(rows)
                for r, v in zip(rows, values):
                    r[metric] = f"{v:.4f}"
                if metric not in header:
                    header.append(metric)
                write_tsv(path, header, rows)
            summary.setdefault(folder, {}).setdefault(pair, {"sentences": len(rows)})[metric] = system
            print(f"  {metric:7s} {folder}/{pair}: {system:.2f}")
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for folder, pairs in summary.items():
        rows = [{"pair": p, **v} for p, v in sorted(pairs.items())]
        avg = {"pair": "average", "sentences": sum(r["sentences"] for r in rows)}
        for m in METRICS:
            vals = [r[m] for r in rows if m in r]
            if vals:
                avg[m] = sum(vals) / len(vals)
        rows.append(avg)
        for r in rows:
            for m in METRICS:
                if m in r:
                    r[m] = f"{r[m]:.2f}"
        path = os.path.join(folder, SUMMARY_NAME)
        write_tsv(path, ["pair", "sentences"] + METRICS, rows)
        print(f"summary -> {path}")


if __name__ == "__main__":
    main()
