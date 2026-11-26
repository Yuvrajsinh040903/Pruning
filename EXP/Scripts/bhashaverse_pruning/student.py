import copy
import re

import torch
from torch import nn

LAYER_KEY = re.compile(r"model\.(encoder|decoder)\.layers\.(\d+)\.(.+)")


def select_layers(n, rule, custom=None):
    if rule == "custom":
        keep = sorted(custom)
    elif rule == "alternate":
        keep = sorted(set(range(0, n, 2)) | {n - 1})
    elif rule == "distilbert":
        k = n // 2
        left = (k + 1) // 2
        keep = [2 * i for i in range(left)] + [n - 1 - 2 * i for i in reversed(range(k - left))]
    else:
        raise ValueError(rule)
    assert len(set(keep)) == len(keep) and all(0 <= i < n for i in keep), keep
    return keep


def kept_layers(config, C):
    enc, dec = config.encoder_layers, config.decoder_layers
    enc_keep = select_layers(enc, C.LAYER_RULE, C.CUSTOM_ENCODER_LAYERS) if C.PRUNE_ENCODER else list(range(enc))
    dec_keep = select_layers(dec, C.LAYER_RULE, C.CUSTOM_DECODER_LAYERS) if C.PRUNE_DECODER else list(range(dec))
    return enc_keep, dec_keep


def build_student(teacher, enc_keep, dec_keep, dropout=None):
    cfg = copy.deepcopy(teacher.config)
    cfg.encoder_layers, cfg.decoder_layers = len(enc_keep), len(dec_keep)
    cfg.num_hidden_layers = cfg.encoder_layers
    if dropout is not None:
        cfg.dropout = dropout
    student = type(teacher)(cfg)

    # Keep the teacher's embedding sharing: encoder/decoder embed_tokens are tied to model.shared only if they are in the teacher
    t, s = teacher.model, student.model
    for name in ["encoder", "decoder"]:
        t_emb, s_emb = getattr(t, name).embed_tokens, getattr(s, name).embed_tokens
        if t_emb.weight is t.shared.weight:
            s_emb.weight = s.shared.weight
        elif s_emb.weight is s.shared.weight:
            s_emb.weight = nn.Parameter(torch.empty_like(s.shared.weight))

    keep = {"encoder": enc_keep, "decoder": dec_keep}
    state = {}
    for key, value in teacher.state_dict().items():
        m = LAYER_KEY.fullmatch(key)
        if m is None:
            state[key] = value
        elif int(m[2]) in keep[m[1]]:
            state[f"model.{m[1]}.layers.{keep[m[1]].index(int(m[2]))}.{m[3]}"] = value
    result = student.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    return student.to(dtype=next(teacher.parameters()).dtype)


def param_count(model):
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total
