import csv
import hashlib
import json
import os

import torch
from comet import download_model, load_from_checkpoint

import config as C
from data import load_pairs

# Run after run.py: scores the saved hypotheses and adds comet_base / comet_pruned to results.csv and every result.json
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 32
GPUS = 1 if torch.cuda.is_available() else 0
CACHE_FILE = os.path.join(C.OUTPUT_DIR, "cache", "comet_scores.json")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().rstrip("\n").split("\n")


def baseline_hyps(model_name, tgt):
    with open(os.path.join(C.OUTPUT_DIR, "cache", model_name, f"baseline_{tgt}.json"), encoding="utf-8") as f:
        return json.load(f)["hyps"]


def key_of(hyps, refs):
    # Scores are cached by content, so re-running only scores new or changed outputs
    h = hashlib.sha1()
    for line in hyps + ["\x00"] + refs:
        h.update(line.encode("utf-8") + b"\n")
    return f"{COMET_MODEL}:{h.hexdigest()}"


class CometScorer:
    def __init__(self):
        self.model = None
        self.cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                self.cache = json.load(f)

    def score(self, sources, hyps, refs):
        key = key_of(hyps, refs)
        if key not in self.cache:
            if self.model is None:
                self.model = load_from_checkpoint(download_model(COMET_MODEL))
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
            out = self.model.predict(data, batch_size=COMET_BATCH_SIZE, gpus=GPUS, progress_bar=True)
            self.cache[key] = float(out.system_score)
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=1)
        return self.cache[key]


def result_files():
    root = os.path.join(C.OUTPUT_DIR, "models")
    for model_name in sorted(os.listdir(root)):
        for exp in sorted(os.listdir(os.path.join(root, model_name))):
            path = os.path.join(root, model_name, exp, "result.json")
            if os.path.exists(path):
                yield os.path.join(root, model_name, exp), path


def write_csv(rows, path):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    scorer = CometScorer()
    evals, all_rows = {}, []
    for exp_dir, path in result_files():
        with open(path) as f:
            rows = json.load(f)
        for row in rows:
            tgt = row["tgt"]
            if tgt not in evals:
                evals[tgt] = load_pairs(C.EVAL_SPLIT, tgt, C.MAX_EVAL_LINES)
            sources, refs = evals[tgt]
            pruned = read_lines(os.path.join(exp_dir, f"hyps.{tgt}.txt"))
            base = baseline_hyps(row["model"], tgt)
            assert len(pruned) == len(base) == len(refs), f"{exp_dir} {tgt}: hypothesis and reference counts differ"
            row["comet_base"] = scorer.score(sources, base, refs)
            row["comet_pruned"] = scorer.score(sources, pruned, refs)
            print(f"{row['model']} {row['setting']} {row['mode']} {row['group']} {tgt}: "
                  f"COMET {row['comet_base']:.4f} -> {row['comet_pruned']:.4f}")
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        all_rows.extend(rows)
    write_csv(all_rows, os.path.join(C.OUTPUT_DIR, "results.csv"))


if __name__ == "__main__":
    main()