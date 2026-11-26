import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM

PROMPT = """Translate the following sentence from English to {lang}:
{text}
RESPONSE FORMAT: <ONLY TRANSLATED TEXT>"""

RESERVED = re.compile(r"<\|reserved_special_token_\d+\|>")
DIGITS = re.compile(r"\d{1,3}")


def read_pairs(path, src_col, tgt_col, limit=None):
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, on_bad_lines="skip")
    pairs = list(df[[src_col, tgt_col]].itertuples(index=False, name=None))
    return pairs[:limit] if limit else pairs


def render(tok, lang, src, tgt=None):
    msgs = [{"role": "user", "content": PROMPT.format(lang=lang, text=src)}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return text if tgt is None else text + tgt + tok.eos_token


def build_corpus(tok, langs, template, src_col, tgt_col, limit=None):
    corpus = {}
    for code, name in langs.items():
        corpus[code] = [render(tok, name, s, t) for s, t in read_pairs(template.format(tgt=code), src_col, tgt_col, limit)]
    return corpus


class BPE:
    def __init__(self, tok):
        state = json.loads(tok.backend_tokenizer.to_str())
        model = state["model"]
        assert model["type"] == "BPE", model["type"]
        self.vocab = model["vocab"]
        self.ranks = {tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m): i for i, m in enumerate(model["merges"])}
        self.ignore_merges = model.get("ignore_merges", False)
        self.pre = tok.backend_tokenizer.pre_tokenizer
        self.special = {t["content"]: t["id"] for t in state["added_tokens"]}
        self.split = re.compile("(" + "|".join(map(re.escape, sorted(self.special, key=len, reverse=True))) + ")")
        self.cache = {}

    def word(self, w):
        if w in self.cache:
            return self.cache[w]
        if self.ignore_merges and w in self.vocab:
            out = ([w], set())
        else:
            parts, seen = list(w), set()
            while len(parts) > 1:
                best = min(((self.ranks[p], i) for i, p in enumerate(zip(parts, parts[1:])) if p in self.ranks), default=None)
                if best is None:
                    break
                i = best[1]
                parts[i:i + 2] = [parts[i] + parts[i + 1]]
                seen.add(parts[i])
            out = (parts, seen)
        self.cache[w] = out
        return out

    def words(self, text):
        for chunk in self.split.split(text):
            if chunk in self.special:
                yield chunk, True
            elif chunk:
                for w, _ in self.pre.pre_tokenize_str(chunk):
                    yield w, False

    def encode(self, text):
        ids = []
        for w, special in self.words(text):
            ids += [self.special[w]] if special else [self.vocab[t] for t in self.word(w)[0]]
        return ids


def select_vocab(tok, corpus, min_count=1, keep_digits=True, extra_texts=()):
    bpe = BPE(tok)
    words, specials = Counter(), Counter()
    for texts in list(corpus.values()) + [list(extra_texts)]:
        for text in texts:
            for w, special in bpe.words(text):
                (specials if special else words)[w] += 1

    counts = Counter()
    for w, c in words.items():
        for t in bpe.word(w)[0]:
            counts[t] += c
    keep = {t for t, c in counts.items() if c >= min_count}
    for w in words:
        final, inter = bpe.word(w)
        if all(t in keep for t in final):
            keep |= inter
    keep |= {t for t in bpe.vocab if len(t) == 1}
    if keep_digits:
        keep |= {t for t in bpe.vocab if DIGITS.fullmatch(t)}

    ids = {bpe.vocab[t] for t in keep}
    ids |= {i for s, i in bpe.special.items() if not RESERVED.fullmatch(s)}
    ids |= {bpe.special[s] for s in specials}
    stats = {"unique_words": len(words), "tokens_seen": len(counts), "tokens_kept_by_count": sum(c >= min_count for c in counts.values())}
    return sorted(ids), bpe, stats


def check_simulation(tok, bpe, texts):
    bad = sum(tok(t, add_special_tokens=False)["input_ids"] != bpe.encode(t) for t in texts)
    assert bad == 0, f"BPE simulation differs from the tokenizer on {bad}/{len(texts)} texts"


def remap_special_ids(node, remap):
    if isinstance(node, dict):
        if "special_tokens" in node:
            for v in node["special_tokens"].values():
                v["ids"] = [remap[i] for i in v["ids"]]
        for v in node.values():
            remap_special_ids(v, remap)
    elif isinstance(node, list):
        for v in node:
            remap_special_ids(v, remap)


def prune_tokenizer(tok, kept_ids, out_dir):
    state = json.loads(tok.backend_tokenizer.to_str())
    remap = {o: n for n, o in enumerate(kept_ids)}
    model = state["model"]
    model["vocab"] = {t: remap[i] for t, i in model["vocab"].items() if i in remap}
    kept = set(model["vocab"])
    pairs = lambda m: m.split(" ", 1) if isinstance(m, str) else m
    model["merges"] = [m for m in model["merges"] if (lambda a, b: a in kept and b in kept and a + b in kept)(*pairs(m))]
    state["added_tokens"] = [dict(t, id=remap[t["id"]]) for t in state["added_tokens"] if t["id"] in remap]
    remap_special_ids(state.get("post_processor"), remap)

    out_dir = Path(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "tokenizer.json").write_text(json.dumps(state, ensure_ascii=False))
    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if "added_tokens_decoder" in cfg:
        cfg["added_tokens_decoder"] = {str(remap[int(k)]): v for k, v in cfg["added_tokens_decoder"].items() if int(k) in remap}
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return remap


def remap_token_ids(cfg, remap):
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        v = getattr(cfg, key, None)
        if isinstance(v, list):
            setattr(cfg, key, [remap[i] for i in v if i in remap])
        elif v is not None:
            setattr(cfg, key, remap.get(v))


def prune_model(model, kept_ids, remap):
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    tied = head.weight.data_ptr() == emb.weight.data_ptr()
    idx = torch.tensor(kept_ids, device=emb.weight.device)
    new_emb = torch.nn.Embedding(len(kept_ids), emb.weight.shape[1], device=emb.weight.device, dtype=emb.weight.dtype)
    new_emb.weight.data = emb.weight.data[idx].clone()
    model.set_input_embeddings(new_emb)
    if tied:
        head.weight = new_emb.weight
    else:
        head.weight = torch.nn.Parameter(head.weight.data[idx.to(head.weight.device)].clone())
        if head.bias is not None:
            head.bias = torch.nn.Parameter(head.bias.data[idx.to(head.bias.device)].clone())
    head.out_features = len(kept_ids)
    model.config.vocab_size = len(kept_ids)
    remap_token_ids(model.config, remap)
    if model.generation_config is not None:
        remap_token_ids(model.generation_config, remap)
    return tied


@torch.no_grad()
def logits_before(model, tok, texts, kept_ids, max_len):
    idx = torch.tensor(kept_ids, device=model.device)
    out = []
    for text in texts:
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_len].to(model.device)
        logits = model(ids).logits[0].float()
        lp = logits.log_softmax(-1)
        removed_mass = 1 - lp[:, idx].exp().sum(-1)
        argmax_removed = ~torch.isin(logits.argmax(-1), idx)
        out.append((logits[:, idx].cpu(), removed_mass.cpu(), argmax_removed.cpu()))
    return out


@torch.no_grad()
def compare_logits(model, new_tok, texts, before, max_len):
    diffs, agree, total, mass, top_removed = [], 0, 0, [], 0
    for text, (ref, rm, ar) in zip(texts, before):
        ids = new_tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_len].to(model.device)
        new = model(ids).logits[0].float().cpu()
        assert new.shape == ref.shape, (new.shape, ref.shape)
        diffs.append((new - ref).abs().max().item())
        agree += (new.argmax(-1) == ref.argmax(-1)).sum().item()
        total += new.shape[0]
        mass.append(rm)
        top_removed += ar.sum().item()
    mass = torch.cat(mass)
    return {"logit_max_abs_diff": max(diffs), "argmax_agreement": agree / total, "removed_mass_mean": mass.mean().item(),
            "removed_mass_max": mass.max().item(), "top1_removed_rate": top_removed / total, "positions": total}


def tokenization_changes(old_tok, new_tok, kept_ids, texts):
    keep = set(kept_ids)
    changed, oov, n = 0, 0, 0
    for t in texts:
        a = old_tok(t, add_special_tokens=False)["input_ids"]
        b = [kept_ids[i] for i in new_tok(t, add_special_tokens=False)["input_ids"]]
        changed += a != b
        oov += sum(i not in keep for i in a)
        n += len(a)
    return {"sentences_changed": changed / max(len(texts), 1), "token_oov_rate": oov / max(n, 1)}


def coverage_report(old_tok, new_tok, kept_ids, langs, template, src_col, tgt_col, splits):
    rows = []
    for split in splits:
        for code, name in langs.items():
            path = Path(template.format(split=split, tgt=code))
            if not path.exists():
                continue
            pairs = read_pairs(path, src_col, tgt_col)
            for side, texts in [("prompt", [render(old_tok, name, s) for s, _ in pairs]), ("reference", [t for _, t in pairs])]:
                rows.append({"split": split, "tgt": code, "side": side, **tokenization_changes(old_tok, new_tok, kept_ids, texts)})
    return pd.DataFrame(rows)


def script_of(text):
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            return name.split(" ")[0] if name else "UNKNOWN"
    return "NON-LETTER"


def kept_by_script(tok, kept_ids):
    total, kept = Counter(), Counter()
    keep = set(kept_ids)
    for i in range(len(tok)):
        s = script_of(tok.decode([i]))
        total[s] += 1
        kept[s] += i in keep
    df = pd.DataFrame({"script": list(total), "tokens": [total[s] for s in total], "kept": [kept[s] for s in total]})
    df["kept_pct"] = 100 * df["kept"] / df["tokens"]
    return df.sort_values("tokens", ascending=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def meta_model(name_or_cfg):
    cfg = AutoConfig.from_pretrained(name_or_cfg) if isinstance(name_or_cfg, (str, Path)) else name_or_cfg
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(cfg)


def budget(base_name, pruned_cfg, new_vocab):
    base = meta_model(base_name)
    base_params, per_layer = count_params(base), count_params(base.model.layers[0])
    layered = meta_model(pruned_cfg)
    layered_params = count_params(layered)
    emb = layered.get_input_embeddings().weight
    head = layered.get_output_embeddings().weight
    tied = pruned_cfg.tie_word_embeddings
    vocab_removed = (pruned_cfg.vocab_size - new_vocab) * emb.shape[1] * (1 if tied else 2)
    final = layered_params - vocab_removed
    return {
        "base_params": base_params, "base_layers": base.config.num_hidden_layers, "params_per_layer": per_layer,
        "layers": pruned_cfg.num_hidden_layers, "layers_removed": base.config.num_hidden_layers - pruned_cfg.num_hidden_layers,
        "params_after_layer_pruning": layered_params, "vocab_before": pruned_cfg.vocab_size, "vocab_after": new_vocab,
        "tied_embeddings": tied, "vocab_params_removed": vocab_removed, "params_final": final,
        "pct_removed_by_layers": 100 * (base_params - layered_params) / base_params,
        "pct_removed_by_vocab": 100 * vocab_removed / base_params,
        "pct_removed_total": 100 * (base_params - final) / base_params,
        "vocab_pruning_in_layers": vocab_removed / per_layer,
    }


def budget_table(info, max_layers):
    rows = []
    for k in range(max_layers + 1):
        layers_only = 100 * k * info["params_per_layer"] / info["base_params"]
        rows.append({"layers_removed": k, "pct_layers_only": layers_only, "pct_with_vocab": layers_only + info["pct_removed_by_vocab"]})
    return pd.DataFrame(rows)
