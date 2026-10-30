import os

from datasets import load_dataset

from config import DATA_TEMPLATE, LANGS, SRC_COLUMN, SRC_LANG, TGT_COLUMN, TGT_LANGS
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

DATA_TEMPLATE = "/home/priyanka/Pruning/EXP/Data/flores_data/{split}/eng-{tgt}.tsv"
# Gated dataset: accept the terms on the dataset page and run `huggingface-cli login` first
FLORES_REPO = "openlanguagedata/flores_plus"
SPLITS = {"devtest": "test", "dev": "dev"}


def flores_lines(lang, split):
    ds = load_dataset(FLORES_REPO, LANGS[lang]["flores"], split=split).sort("id")
    return ds["id"], ds["text"]


def clean(text):
    return " ".join(text.replace("\t", " ").split())


def write_tsv(path, src_lines, tgt_lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{SRC_COLUMN}\t{TGT_COLUMN}\n")
        f.writelines(f"{clean(s)}\t{clean(t)}\n" for s, t in zip(src_lines, tgt_lines))


for flores_split, split in SPLITS.items():
    src_ids, src_lines = flores_lines(SRC_LANG, flores_split)
    for tgt in TGT_LANGS:
        tgt_ids, tgt_lines = flores_lines(tgt, flores_split)
        assert src_ids == tgt_ids, f"{tgt}/{flores_split}: sentence ids are not aligned"
        write_tsv(DATA_TEMPLATE.format(split=split, tgt=tgt), src_lines, tgt_lines)
        print(f"{flores_split} -> {split}: eng-{tgt} {len(src_lines)} pairs")
