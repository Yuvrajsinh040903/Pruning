import json
import os
import shutil

import sentencepiece as spm
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import MBartForConditionalGeneration

TOKENIZER_FILES = ["fairseq_dict.json", "onemtv3b_spm.model"]


def resolve(path, filename):
    local = os.path.join(path, filename)
    return local if os.path.isdir(path) else hf_hub_download(path, filename)


class Codec:
    # "###src-to-tgt### text" -> SentencePiece pieces -> fairseq dictionary ids (not SPM ids)
    def __init__(self, path, flores_codes, max_src_len, max_tgt_len):
        with open(resolve(path, "fairseq_dict.json"), encoding="utf-8") as f:
            fs = json.load(f)
        self.src_vocab, self.tgt_vocab = fs["src"], fs["tgt"]
        self.tgt_id2sym = {v: k for k, v in self.tgt_vocab.items()}
        sp = fs["special"]
        self.bos, self.pad, self.eos, self.unk = sp["bos"], sp["pad"], sp["eos"], sp["unk"]
        self.sp = spm.SentencePieceProcessor(model_file=resolve(path, "onemtv3b_spm.model"))
        self.codes = flores_codes
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len

    def encode_src(self, text, src, tgt):
        tagged = f"###{self.codes[src]}-to-{self.codes[tgt]}### {text}"
        ids = [self.src_vocab.get(p, self.unk) for p in self.sp.encode(tagged, out_type=str)]
        return ids[:self.max_src_len - 1] + [self.eos]

    def encode_tgt(self, text):
        ids = [self.tgt_vocab.get(p, self.unk) for p in self.sp.encode(text, out_type=str)]
        return ids[:self.max_tgt_len - 1] + [self.eos]

    def decode(self, ids):
        pieces = [self.tgt_id2sym.get(t, "<unk>") for t in ids if t not in (self.bos, self.pad, self.eos)]
        return self.sp.decode(pieces).strip()


def checkpoint_keys(path):
    try:
        with safe_open(resolve(path, "model.safetensors"), "pt") as f:
            return set(f.keys())
    except Exception:
        return None


def load_model(path, dtype):
    model = MBartForConditionalGeneration.from_pretrained(path, torch_dtype=dtype)
    # A checkpoint that stores only model.shared can load encoder/decoder embeddings as random on transformers 4.45
    keys = checkpoint_keys(path)
    if keys and "model.shared.weight" in keys:
        for key, module in [("model.encoder.embed_tokens.weight", model.model.encoder.embed_tokens),
                            ("model.decoder.embed_tokens.weight", model.model.decoder.embed_tokens)]:
            if key not in keys and module.weight is not model.model.shared.weight:
                print(f"  tying {key} to model.shared.weight")
                module.weight = model.model.shared.weight
    # Checkpoints saved during gradient-checkpointed training carry use_cache=False, which slows generation
    model.config.use_cache = True
    return model.eval()


def save_model(model, out_dir, tokenizer_src, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    for name in TOKENIZER_FILES:
        shutil.copy(resolve(tokenizer_src, name), os.path.join(out_dir, name))
    if meta is not None:
        with open(os.path.join(out_dir, "prune_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
