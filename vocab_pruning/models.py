import json
import os

import transformers
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tokenization import BhashaverseCodec, NllbCodec

TF_MAJOR, TF_MINOR = (int(x) for x in transformers.__version__.split(".")[:2])


def dtype_kwargs(dtype):
    return {"dtype": dtype} if (TF_MAJOR, TF_MINOR) >= (4, 56) else {"torch_dtype": dtype}


def resolve(path, filename):
    local = os.path.join(path, filename)
    return local if os.path.isdir(path) else hf_hub_download(path, filename)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def checkpoint_keys(path):
    try:
        with safe_open(resolve(path, "model.safetensors"), "pt") as f:
            return set(f.keys())
    except Exception:
        return None


def load_seq2seq(path, dtype):
    model = AutoModelForSeq2SeqLM.from_pretrained(path, **dtype_kwargs(dtype))
    # If the checkpoint stores only model.shared, encoder/decoder embeddings must point to it; some transformers
    # versions otherwise leave them randomly initialised without reporting them as missing
    keys = checkpoint_keys(path)
    if keys and "model.shared.weight" in keys:
        for key, module in [("model.encoder.embed_tokens.weight", model.get_encoder().embed_tokens),
                            ("model.decoder.embed_tokens.weight", model.get_decoder().embed_tokens)]:
            if key not in keys and module.weight is not model.model.shared.weight:
                print(f"  tying {key} to model.shared.weight (not stored in checkpoint)")
                module.weight = model.model.shared.weight
    return model.eval()


def load_bhashaverse(path, langs, dtype):
    model = load_seq2seq(path, dtype)
    with open(resolve(path, "fairseq_dict.json"), encoding="utf-8") as f:
        fs = json.load(f)
    specials = {k: fs["special"][k] for k in ["bos", "pad", "eos", "unk"]}
    codec = BhashaverseCodec(read_bytes(resolve(path, "onemtv3b_spm.model")), fs["src"], fs["tgt"], specials, langs)
    return model, codec


def load_nllb(path, langs, dtype):
    model = load_seq2seq(path, dtype)
    tok = AutoTokenizer.from_pretrained(path)
    vocab = tok.get_vocab()
    specials = {"bos": vocab["<s>"], "pad": vocab["<pad>"], "eos": vocab["</s>"], "unk": vocab["<unk>"]}
    codec = NllbCodec(read_bytes(resolve(path, "sentencepiece.bpe.model")), vocab, dict(vocab), specials, langs)
    check_nllb_codec(codec, tok)
    return model, codec


def check_nllb_codec(codec, tok):
    samples = ["The government announced new measures on Tuesday.", "Prices rose by 3.5% in 2024, said Dr. Rao.",
               "COVID-19 vaccines were distributed across India."]
    tok.src_lang = codec.flores("eng")
    mismatches = sum(tok(s).input_ids != codec.encode_src(s, "eng", "hin") for s in samples)
    if mismatches:
        print(f"  WARNING: NLLB codec differs from the HF tokenizer on {mismatches}/{len(samples)} samples")


LOADERS = {
    "bhashaverse": load_bhashaverse,
    "nllb": load_nllb,
}


def load(name, path, langs, dtype, device):
    model, codec = LOADERS[name](path, langs, dtype)
    return model.to(device), codec
