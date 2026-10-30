import os
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
from datasets import get_dataset_split_names, load_dataset

# from config import LANGS, SRC_COLUMN, TGT_COLUMN, TGT_LANGS
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

DATA_TEMPLATE = "data/{split}/eng-{tgt}.tsv"

# Gated dataset: accept the terms on https://huggingface.co/datasets/ai4bharat/BPCC and run `huggingface-cli login` first
BPCC_REPO = "ai4bharat/BPCC"
OUT_DIR = "/home/priyanka/Pruning/EXP/Data/bpcc_data/train"

# BPCC-Human subsets (~100-300K pairs per language). Mined subsets ("samanantar++", "nllb-filtered",
# "samanantar-filtered", "comparable") add millions of pairs per language; cap them with MAX_PAIRS_PER_SUBSET.
SUBSETS = ["bpcc-seed-latest", "daily", "ilci", "massive"]
MAX_PAIRS_PER_SUBSET = None
MAX_PAIRS_PER_LANG = None

# Pairs whose source or target also appears in these files are dropped, so token selection never sees test sentences
EXCLUDE_TEMPLATE = "/home/priyanka/Pruning/EXP/Data/bpcc_data/test/eng-{tgt}.tsv"


def clean(text):
    return " ".join((text or "").replace("\t", " ").split())


def load_exclusions(tgt):
    path = EXCLUDE_TEMPLATE.format(tgt=tgt)
    if not os.path.exists(path):
        print(f"  no exclusion file {path}, test overlap is not removed")
        return set()
    with open(path, encoding="utf-8") as f:
        next(f)
        return {clean(part) for line in f for part in line.rstrip("\n").split("\t")[:2]}


def split_name(subset, flores_code):
    # Split names are not always cased the same across subsets (e.g. "ben_beng" in "massive")
    names = {n.lower(): n for n in get_dataset_split_names(BPCC_REPO, subset)}
    return names.get(flores_code.lower())


def iter_pairs(subset, split, src_code):
    ds = load_dataset(BPCC_REPO, subset, split=split, streaming=True)
    for row in ds:
        # Each split stores eng -> indic pairs; swap if a subset stores the other direction
        if row.get("src_lang") == src_code:
            yield clean(row["src"]), clean(row["tgt"])
        elif row.get("tgt_lang") == src_code:
            yield clean(row["tgt"]), clean(row["src"])


def download_pair(tgt):
    flores = LANGS[tgt]["flores"]
    excluded = load_exclusions(tgt)
    seen, pairs = set(), []
    for subset in SUBSETS:
        split = split_name(subset, flores)
        if split is None:
            print(f"  {subset}: no {flores} split, skipped")
            continue
        added = 0
        for src, trg in iter_pairs(subset, split, LANGS["eng"]["flores"]):
            if not src or not trg or (src, trg) in seen or src in excluded or trg in excluded:
                continue
            seen.add((src, trg))
            pairs.append((src, trg))
            added += 1
            if MAX_PAIRS_PER_SUBSET and added >= MAX_PAIRS_PER_SUBSET:
                break
            if MAX_PAIRS_PER_LANG and len(pairs) >= MAX_PAIRS_PER_LANG:
                break
        print(f"  {subset}/{split}: {added} pairs")
        if MAX_PAIRS_PER_LANG and len(pairs) >= MAX_PAIRS_PER_LANG:
            break
    return pairs


def write_tsv(path, pairs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{SRC_COLUMN}\t{TGT_COLUMN}\n")
        f.writelines(f"{s}\t{t}\n" for s, t in pairs)


if __name__ == "__main__":
    for tgt in TGT_LANGS:
        print(f"eng-{tgt}")
        pairs = download_pair(tgt)
        path = os.path.join(OUT_DIR, f"eng-{tgt}.tsv")
        write_tsv(path, pairs)
        print(f"  wrote {len(pairs)} pairs to {path}")