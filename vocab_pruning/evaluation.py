import time

import sacrebleu
import torch

from config import BLEU_TOKENIZE


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def pad_batch(seqs, pad, device):
    ids = torch.full((len(seqs), max(map(len, seqs))), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def model_device(model):
    return next(model.parameters()).device


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


def allocated_mb(device):
    return torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None


@torch.no_grad()
def translate(model, codec, sources, src, tgt, gen_kwargs, batch_size):
    device = model_device(model)
    enc = [codec.encode_src(s, src, tgt) for s in sources]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]), reverse=True)
    kwargs = {**gen_kwargs, **codec.gen_kwargs(tgt)}
    out = [None] * len(enc)
    reset_peak(device)
    start = time.perf_counter()
    for batch in chunks(order, batch_size):
        ids, mask = pad_batch([enc[i] for i in batch], codec.pad, device)
        generated = model.generate(input_ids=ids, attention_mask=mask, **kwargs)
        for i, g in zip(batch, generated.tolist()):
            out[i] = g
    perf = {"seconds": time.perf_counter() - start, "peak_MB": peak_mb(device)}
    return [codec.decode(g) for g in out], out, perf


def score(hyps, refs):
    return {"bleu": sacrebleu.corpus_bleu(hyps, [refs], tokenize=BLEU_TOKENIZE).score,
            "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score}


@torch.no_grad()
def teacher_forced(model, enc, labels, start_id, pad, batch_size, fn):
    # fn(sentence_index, logits[T, V]) is called with the logits at every label position of each sentence
    device = model_device(model)
    for batch in chunks(list(range(len(enc))), batch_size):
        ids, mask = pad_batch([enc[i] for i in batch], pad, device)
        dec, dec_mask = pad_batch([[start_id] + labels[i][:-1] for i in batch], pad, device)
        logits = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec, decoder_attention_mask=dec_mask).logits
        for row, i in enumerate(batch):
            fn(i, logits[row, :len(labels[i])])


def coverage(codec, sources, refs, src, tgt, keep_src, keep_tgt):
    ks, kt = set(keep_src), set(keep_tgt)
    enc = [codec.encode_src(s, src, tgt) for s in sources]
    lab = [codec.encode_tgt(r, tgt) for r in refs]
    src_tokens = sum(map(len, enc))
    tgt_tokens = sum(map(len, lab))
    return {"src_oov_rate": sum(i not in ks for e in enc for i in e) / src_tokens,
            "ref_oov_rate": sum(i not in kt for l in lab for i in l) / tgt_tokens,
            "src_sent_changed": sum(not set(e) <= ks for e in enc) / len(enc)}


def fidelity_before(model, codec, sources, refs, src, tgt, keep_src, keep_tgt, n, batch_size):
    # Original model: probability mass on removed tokens, plus kept-token logits of fully in-vocabulary sentences for the later comparison
    sources, refs = sources[:n], refs[:n]
    enc = [codec.encode_src(s, src, tgt) for s in sources]
    lab = [codec.encode_tgt(r, tgt) for r in refs]
    ks, kt = set(keep_src), set(keep_tgt)
    exact = {i for i in range(len(enc)) if set(enc[i]) <= ks and set(lab[i]) <= kt}
    keep_idx = torch.tensor(keep_tgt, device=model_device(model))
    state = {"exact": exact, "stored": {}, "removed_mass": [], "argmax_removed": 0, "positions": 0,
             "sources": sources, "refs": refs}

    def fn(i, logits):
        logp = logits.float().log_softmax(-1)
        removed = 1 - logp.index_select(-1, keep_idx).logsumexp(-1).exp()
        state["removed_mass"].extend(removed.clamp(min=0).tolist())
        state["argmax_removed"] += (~torch.isin(logp.argmax(-1), keep_idx)).sum().item()
        state["positions"] += logits.shape[0]
        if i in exact:
            state["stored"][i] = logits.index_select(-1, keep_idx).float().cpu()

    start = codec.gen_kwargs(tgt)["decoder_start_token_id"]
    teacher_forced(model, enc, lab, start, codec.pad, batch_size, fn)
    return state


def fidelity_after(model, codec, src, tgt, state, batch_size):
    idx = sorted(state["exact"])
    enc = [codec.encode_src(state["sources"][i], src, tgt) for i in idx]
    lab = [codec.encode_tgt(state["refs"][i], tgt) for i in idx]
    diffs, agree, total = [0.0], 0, 0

    def fn(j, logits):
        nonlocal agree, total
        ref = state["stored"][idx[j]].to(logits.device)
        logits = logits.float()
        diffs.append((logits - ref).abs().max().item())
        agree += (logits.argmax(-1) == ref.argmax(-1)).sum().item()
        total += logits.shape[0]

    start = codec.gen_kwargs(tgt)["decoder_start_token_id"]
    teacher_forced(model, enc, lab, start, codec.pad, batch_size, fn)
    mass = state["removed_mass"]
    return {"exact_sentences": len(idx), "fidelity_sentences": len(state["sources"]),
            "logit_max_abs_diff": max(diffs), "argmax_agreement": agree / max(total, 1),
            "removed_mass_mean": sum(mass) / max(len(mass), 1), "removed_mass_max": max(mass, default=0.0),
            "argmax_removed_rate": state["argmax_removed"] / max(state["positions"], 1)}
