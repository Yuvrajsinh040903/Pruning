import csv
import os
import sys

from config import DATA_TEMPLATE, SRC_COLUMN, TGT_COLUMN

csv.field_size_limit(sys.maxsize)


def read_tsv_pairs(path, limit=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing data file: {path} (edit DATA_TEMPLATE in config.py)")
    src_lines, tgt_lines = [], []
    with open(path, encoding="utf-8", newline="") as f:
        # QUOTE_NONE so quote characters inside sentences are kept as text
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = {SRC_COLUMN, TGT_COLUMN} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}, found {reader.fieldnames}")
        for row in reader:
            src, tgt = row[SRC_COLUMN], row[TGT_COLUMN]
            if src is None or tgt is None:
                continue
            src_lines.append(src.strip())
            tgt_lines.append(tgt.strip())
            if limit and len(src_lines) >= limit:
                break
    return src_lines, tgt_lines


def load_pairs(split, tgt, limit=None):
    return read_tsv_pairs(DATA_TEMPLATE.format(split=split, tgt=tgt), limit)
