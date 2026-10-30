import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from contextlib import contextmanager
from datasets import load_dataset
from torch.cuda import empty_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """Translate the following sentence from English to {lang}:
{text}
RESPONSE FORMAT: <ONLY TRANSLATED TEXT>"""

LANG_NAMES = {
    "hin_Deva": "Hindi", "tel_Telu": "Telugu", "guj_Gujr": "Gujarati", "tam_Taml": "Tamil",
    "ben_Beng": "Bengali", "urd_Arab": "Urdu", "kan_Knda": "Kannada",
}

MAX_ROWS = 4000
DEBIASED_CKA = True
NORMALIZER = "zscore"
SCOPES = {"cosine": "all", "cka": "all"}

# global: pruned vs current model (last layer), local: layer input vs output
# weight applies to the global term; local gets 1 - weight
STRATEGIES = {
    "cosine": {"global": "cosine", "local": None, "weight": 1.0},
    "cka": {"global": "cka", "local": None, "weight": 1.0},
    "cos_cka": {"global": "cosine", "local": "cka", "weight": 0.5},
    "cka_cos": {"global": "cka", "local": "cosine", "weight": 0.5},
}


def cosine_sim(x, y):
    return F.cosine_similarity(x.float(), y.float(), dim=-1).mean().item()


def center_gram(gram, debiased):
    gram = gram.to(torch.float64, copy=True)
    n = gram.shape[0]
    if debiased:
        gram.fill_diagonal_(0)
        means = gram.sum(0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        gram -= means[:, None] + means[None, :]
        gram.fill_diagonal_(0)
    else:
        means = gram.mean(0)
        means -= means.mean() / 2
        gram -= means[:, None] + means[None, :]
    return gram


def linear_cka(x, y, debiased=DEBIASED_CKA):
    x, y = x.float(), y.float()
    gx = center_gram(x @ x.T, debiased)
    gy = center_gram(y @ y.T, debiased)
    return ((gx * gy).sum() / (gx.norm() * gy.norm())).item()


METRICS = {"cosine": cosine_sim, "cka": linear_cka}

NORMALIZERS = {
    "zscore": lambda s: (s - s.mean()) / (s.std() + 1e-8),
    "rank": lambda s: s.rank(pct=True),
    "minmax": lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8),
}


class Calibration:
    def __init__(self, inputs, device, max_rows, seed=0):
        self.inputs = inputs.to(device)
        self.mask = self.inputs["attention_mask"].bool()
        gen = torch.Generator().manual_seed(seed)
        n = int(self.mask.sum())
        self.row_idx = torch.randperm(n, generator=gen)[:max_rows].sort().values

    def rows(self, x, scope):
        if scope == "last":
            return x[:, -1, :]
        return x[self.mask.to(x.device)][self.row_idx.to(x.device)]


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name):
    return AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype=torch.float16, attn_implementation="eager"
    )


def build_multilingual_inputs(tokenizer, sources, src_col, num_samples):
    messages = []
    for path, lang in sources:
        ds = load_dataset("csv", data_files=path, delimiter="\t", split="train").select(range(num_samples))
        messages += [[{"role": "user", "content": PROMPT.format(lang=lang, text=s)}] for s in ds[src_col]]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True,
    )


def build_inputs(tokenizer, dataset_path, num_samples):
    return build_multilingual_inputs(tokenizer, [(dataset_path, "Hindi")], "src", num_samples)


def reindex_layers(model):
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = idx
    model.config.num_hidden_layers = len(model.model.layers)


@contextmanager
def without_layer(model, idx):
    layers = list(model.model.layers)
    model.model.layers = nn.ModuleList(layers[:idx] + layers[idx + 1:])
    reindex_layers(model)
    try:
        yield
    finally:
        model.model.layers = nn.ModuleList(layers)
        reindex_layers(model)


def drop_layer(model, idx):
    layers = list(model.model.layers)
    removed = layers.pop(idx)
    model.model.layers = nn.ModuleList(layers)
    reindex_layers(model)
    return removed


def final_states(model, calib):
    out = model(**calib.inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1].clone()


def layer_io_scores(model, calib, scopes):
    scores = [None] * len(model.model.layers)

    def make_hook(i):
        def hook(module, args, kwargs, output):
            x_in = args[0] if args else kwargs["hidden_states"]
            x_out = output[0] if isinstance(output, tuple) else output
            scores[i] = {
                f"l_{m}": fn(calib.rows(x_in, scopes[m]), calib.rows(x_out, scopes[m]))
                for m, fn in METRICS.items()
            }
        return hook

    handles = [
        layer.register_forward_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        model(**calib.inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return scores


@torch.inference_mode()
def candidate_table(model, calib, layer_ids, scopes=SCOPES):
    ref = final_states(model, calib)
    local = layer_io_scores(model, calib, scopes)
    rows = []
    for i, layer_id in enumerate(layer_ids):
        with without_layer(model, i):
            pruned = final_states(model, calib)
        glob = {
            f"g_{m}": fn(calib.rows(ref, scopes[m]), calib.rows(pruned, scopes[m]))
            for m, fn in METRICS.items()
        }
        rows.append({"layer_id": layer_id, **glob, **local[i]})
    empty_cache()
    return pd.DataFrame(rows)


def combined_score(table, strategy, normalizer=NORMALIZER):
    norm = NORMALIZERS[normalizer]
    w = strategy["weight"]
    score = w * norm(table[f"g_{strategy['global']}"])
    if strategy["local"]:
        score = score + (1 - w) * norm(table[f"l_{strategy['local']}"])
    return score


def select_layer(model, calib, active_ids, strategy, normalizer=NORMALIZER, scopes=SCOPES):
    table = candidate_table(model, calib, active_ids, scopes)
    table["score"] = combined_score(table, strategy, normalizer)
    assert not table["score"].isna().any(), "NaN in layer scores"
    best = int(table["score"].values.argmax())
    table["selected"] = table.index == best
    return best, table


def prune_step(model, calib, active_ids, strategy, normalizer=NORMALIZER, scopes=SCOPES):
    best, table = select_layer(model, calib, active_ids, strategy, normalizer, scopes)
    layer_id = active_ids.pop(best)
    drop_layer(model, best)
    return layer_id, table