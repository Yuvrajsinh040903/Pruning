import csv
import os
import random
import sys

import torch

import config as C

csv.field_size_limit(sys.maxsize)


def read_tsv_pairs(path, limit=None):
    sources, targets = [], []
    with open(path, encoding="utf-8", newline="") as f:
        # QUOTE_NONE so quote characters inside sentences are kept as text
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = {C.SRC_COLUMN, C.TGT_COLUMN} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}, found {reader.fieldnames}")
        for row in reader:
            src, tgt = row[C.SRC_COLUMN], row[C.TGT_COLUMN]
            if not src or not tgt or not src.strip() or not tgt.strip():
                continue
            sources.append(src.strip())
            targets.append(tgt.strip())
            if limit and len(sources) >= limit:
                break
    return sources, targets


def data_path(split, tgt):
    return C.DATA_TEMPLATE.format(split=split, tgt=tgt)


def load_train_dev(tgt):
    sources, targets = read_tsv_pairs(data_path(C.TRAIN_SPLIT, tgt), C.MAX_TRAIN_PAIRS_PER_LANG)
    dev_path = data_path(C.DEV_SPLIT, tgt)
    if os.path.exists(dev_path):
        return (sources, targets), read_tsv_pairs(dev_path, C.DEV_SIZE)
    print(f"  {dev_path} not found, holding out the last {C.DEV_SIZE} train pairs for eng-{tgt}")
    return (sources[:-C.DEV_SIZE], targets[:-C.DEV_SIZE]), (sources[-C.DEV_SIZE:], targets[-C.DEV_SIZE:])


def load_test(tgt):
    return read_tsv_pairs(data_path(C.TEST_SPLIT, tgt), C.MAX_TEST_PAIRS)


def encode_pairs(codec, sources, targets, tgt):
    return [(codec.encode_src(s, C.SRC_LANG, tgt), codec.encode_tgt(t)) for s, t in zip(sources, targets)]


def token_batches(examples, max_tokens, shuffle, seed=0):
    # Length-sorted batches capped by padded token count (max(src, tgt) length x batch size)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    batches = []
    # Sort inside large chunks so batches are length-homogeneous but the data order stays random
    chunk = max_tokens * 100 if shuffle else len(order)
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: max(len(examples[i][0]), len(examples[i][1])))
        batch, longest = [], 0
        for i in part:
            n = max(len(examples[i][0]), len(examples[i][1]))
            if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
                batches.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, n)
        if batch:
            batches.append(batch)
    if shuffle:
        rng.shuffle(batches)
    return batches


def pad(seqs, value):
    out = torch.full((len(seqs), max(map(len, seqs))), value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate(examples, idx, codec, device):
    src = [examples[i][0] for i in idx]
    tgt = [examples[i][1] for i in idx]
    input_ids = pad(src, codec.pad)
    # fairseq convention: the decoder starts from EOS and predicts target + EOS
    dec_in = pad([[codec.eos] + t[:-1] for t in tgt], codec.pad)
    labels = pad(tgt, -100)
    return {"input_ids": input_ids.to(device),
            "attention_mask": (input_ids != codec.pad).long().to(device),
            "decoder_input_ids": dec_in.to(device),
            "decoder_attention_mask": (labels != -100).long().to(device),
            "labels": labels.to(device)}
