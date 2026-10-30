import json
import os

import torch
from safetensors.torch import load_model, save_model
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig

from tokenization import load_codec

EMBEDDING_PATHS = {
    "shared": "model.shared",
    "encoder": "model.encoder.embed_tokens",
    "decoder": "model.decoder.embed_tokens",
    "lm_head": "lm_head",
}


def embedding_modules(model):
    mods = {}
    for role, path in EMBEDDING_PATHS.items():
        try:
            mods[role] = model.get_submodule(path)
        except AttributeError:
            pass
    return mods


def tie_groups(model):
    groups = {}
    for role, module in embedding_modules(model).items():
        groups.setdefault(module.weight.data_ptr(), []).append(role)
    return list(groups.values())


def plan_groups(model, mode):
    # Each tied group is sliced once. "split" unties the encoder so it can keep only source-side rows.
    plan = []
    for roles in tie_groups(model):
        if mode == "split" and "encoder" in roles and len(set(roles) - {"shared", "encoder"}) > 0:
            plan.append((["encoder"], "src"))
            plan.append(([r for r in roles if r != "encoder"], "tgt"))
        else:
            side = "src" if "encoder" in roles and set(roles) <= {"shared", "encoder"} else "tgt"
            plan.append((roles, side))
    return plan


def set_rows(mods, roles, weight):
    param = nn.Parameter(weight)
    for role in roles:
        module = mods[role]
        module.weight = param
        if isinstance(module, nn.Embedding):
            module.num_embeddings = weight.shape[0]
        else:
            module.out_features = weight.shape[0]


def remap_generation_config(model, keep_tgt):
    new_id = {old: i for i, old in enumerate(keep_tgt)}
    gc = model.generation_config
    for key, value in gc.to_dict().items():
        if key.endswith("token_id") and isinstance(value, int):
            setattr(gc, key, new_id.get(value))


def prune_model(model, keep_src, keep_tgt, mode):
    mods = embedding_modules(model)
    device = next(model.parameters()).device
    ids = {"src": torch.tensor(keep_src, device=device), "tgt": torch.tensor(keep_tgt, device=device)}
    groups = []
    for roles, side in plan_groups(model, mode):
        old = mods[roles[0]].weight.data
        set_rows(mods, roles, old.index_select(0, ids[side]).clone())
        groups.append({"roles": roles, "side": side, "rows": len(keep_src if side == "src" else keep_tgt)})
    lm_head = mods.get("lm_head")
    if lm_head is not None and getattr(lm_head, "bias", None) is not None:
        lm_head.bias = nn.Parameter(lm_head.bias.data.index_select(0, ids["tgt"]).clone())
    if hasattr(model, "final_logits_bias"):
        model.final_logits_bias = model.final_logits_bias.index_select(1, ids["tgt"]).clone()
    model.config.vocab_size = len(keep_tgt)
    remap_generation_config(model, keep_tgt)
    return {"mode": mode, "groups": groups, "vocab_src": len(keep_src), "vocab_tgt": len(keep_tgt)}


def save_pruned(model, codec, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.config.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)
    save_model(model, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "pruning_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    codec.save(out_dir)


def load_pruned(out_dir, dtype=torch.float32, device="cpu"):
    config = AutoConfig.from_pretrained(out_dir)
    model = AutoModelForSeq2SeqLM.from_config(config)
    with open(os.path.join(out_dir, "pruning_meta.json")) as f:
        meta = json.load(f)
    mods = embedding_modules(model)
    dim = next(iter(mods.values())).weight.shape[1]
    for group in meta["groups"]:
        set_rows(mods, group["roles"], torch.empty(group["rows"], dim))
    load_model(model, os.path.join(out_dir, "model.safetensors"), strict=True)
    model.generation_config = GenerationConfig.from_pretrained(out_dir)
    return model.to(device=device, dtype=dtype).eval(), load_codec(out_dir)


def weight_stats(model):
    params = dict(model.named_parameters())
    total = sum(p.numel() for p in params.values())
    nbytes = sum(p.numel() * p.element_size() for p in params.values())
    nbytes += sum(b.numel() * b.element_size() for b in model.buffers())
    emb = {id(m.weight): m.weight.numel() for m in embedding_modules(model).values()}
    return {"params_M": total / 1e6, "embed_params_M": sum(emb.values()) / 1e6, "weights_MB": nbytes / 2**20}
