import csv
import json
import os
import time

import sacrebleu
import torch

import config as C
from data import load_test, pad
from student import param_count


@torch.no_grad()
def translate(model, codec, sources, tgt):
    enc = [codec.encode_src(s, C.SRC_LANG, tgt) for s in sources]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]), reverse=True)
    out = [None] * len(enc)
    for start in range(0, len(order), C.EVAL_BATCH_SIZE):
        batch = order[start:start + C.EVAL_BATCH_SIZE]
        ids = pad([enc[i] for i in batch], codec.pad).to(C.DEVICE)
        generated = model.generate(input_ids=ids, attention_mask=(ids != codec.pad).long(),
                                   decoder_start_token_id=codec.eos, num_beams=C.BEAM_SIZE,
                                   max_new_tokens=C.MAX_NEW_TOKENS, early_stopping=True,
                                   no_repeat_ngram_size=C.NO_REPEAT_NGRAM_SIZE, repetition_penalty=C.REPETITION_PENALTY)
        for i, g in zip(batch, generated.tolist()):
            out[i] = codec.decode(g)
    return out


def write_hyps(path, sources, refs, hyps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(["source", "target", "hypothesis"])
        for row in zip(sources, refs, hyps):
            w.writerow([x.replace("\t", " ").replace("\n", " ") for x in row])


def evaluate_model(model, codec, name, out_dir):
    # Results are cached in out_dir/results.json so a finished stage is not decoded again
    cache = os.path.join(out_dir, "results.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    model.to(C.DEVICE).eval()
    enc_layers, dec_layers, params = model.config.encoder_layers, model.config.decoder_layers, param_count(model)
    rows = []
    for tgt in C.TGT_LANGS:
        sources, refs = load_test(tgt)
        start = time.perf_counter()
        hyps = translate(model, codec, sources, tgt)
        seconds = time.perf_counter() - start
        write_hyps(os.path.join(out_dir, "hyps", f"{C.SRC_LANG}-{tgt}.tsv"), sources, refs, hyps)
        row = {"model": name, "pair": f"{C.SRC_LANG}-{tgt}", "enc_layers": enc_layers, "dec_layers": dec_layers,
               "params": params,
               "bleu": sacrebleu.corpus_bleu(hyps, [refs], tokenize=C.BLEU_TOKENIZE).score,
               "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
               "seconds": seconds}
        print(f"  {name} {row['pair']}: BLEU {row['bleu']:.2f} chrF++ {row['chrf++']:.2f} ({seconds:.0f}s)")
        rows.append(row)
    with open(cache, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def write_results(path, rows):
    keys = list(rows[0].keys())
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for name, rs in by_model.items():
        avg = {k: rs[0][k] for k in keys}
        avg.update({"pair": "avg", **{k: sum(r[k] for r in rs) / len(rs) for k in ["bleu", "chrf++", "seconds"]}})
        rows = rows + [avg]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"results -> {path}")
