import json
import os

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb


def restrict_spm(proto, removed_pieces):
    # Removed pieces become UNUSED, so SentencePiece re-segments them into smaller kept pieces instead of emitting them
    if not removed_pieces:
        return proto
    m = sp_pb.ModelProto()
    m.ParseFromString(proto)
    for p in m.pieces:
        if p.type == sp_pb.ModelProto.SentencePiece.NORMAL and p.piece in removed_pieces:
            p.type = sp_pb.ModelProto.SentencePiece.UNUSED
    return m.SerializeToString()


def remap_vocab(vocab, keep_old_ids):
    new_id = {old: i for i, old in enumerate(keep_old_ids)}
    return {tok: new_id[i] for tok, i in vocab.items() if i in new_id}


class SpmCodec:
    def __init__(self, spm_proto, src_vocab, tgt_vocab, specials, langs, src_proto=None, tgt_proto=None):
        self.spm_proto = spm_proto
        self.src_proto = src_proto or spm_proto
        self.tgt_proto = tgt_proto or spm_proto
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tgt_id2tok = {i: t for t, i in tgt_vocab.items()}
        self.specials = specials
        self.langs = langs
        self.sp_full = spm.SentencePieceProcessor(model_proto=self.spm_proto)
        self.sp_src = spm.SentencePieceProcessor(model_proto=self.src_proto)
        self.sp_tgt = spm.SentencePieceProcessor(model_proto=self.tgt_proto)

    @property
    def unk(self):
        return self.specials["unk"]

    @property
    def eos(self):
        return self.specials["eos"]

    @property
    def pad(self):
        return self.specials["pad"]

    def flores(self, lang):
        return self.langs[lang]["flores"]

    def src_ids(self, pieces):
        return [self.src_vocab.get(p, self.unk) for p in pieces]

    def tgt_ids(self, pieces):
        return [self.tgt_vocab.get(p, self.unk) for p in pieces]

    def encode_src(self, text, src, tgt):
        raise NotImplementedError

    def encode_tgt(self, text, tgt):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def gen_kwargs(self, tgt):
        raise NotImplementedError

    def required_src_ids(self, src, tgt):
        return set(self.encode_src("", src, tgt)) | set(self.specials.values())

    def required_tgt_ids(self, tgt):
        kw = self.gen_kwargs(tgt)
        ids = set(self.encode_tgt("", tgt)) | set(self.specials.values())
        ids |= {v for k, v in kw.items() if k.endswith("token_id") and v is not None}
        return ids

    def pruned(self, keep_src, keep_tgt):
        src_vocab = remap_vocab(self.src_vocab, keep_src)
        tgt_vocab = remap_vocab(self.tgt_vocab, keep_tgt)
        specials = {k: remap_vocab({k: v}, keep_tgt)[k] for k, v in self.specials.items()}
        assert specials == self.specials, "special token ids must not move; keep sets must contain ids 0..3"
        assert all(remap_vocab({k: v}, keep_src)[k] == v for k, v in self.specials.items())
        src_proto = restrict_spm(self.spm_proto, set(self.src_vocab) - set(src_vocab))
        tgt_proto = restrict_spm(self.spm_proto, set(self.tgt_vocab) - set(tgt_vocab))
        return type(self)(self.spm_proto, src_vocab, tgt_vocab, specials, self.langs, src_proto, tgt_proto)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name, proto in [("spm_full.model", self.spm_proto), ("spm_src.model", self.src_proto), ("spm_tgt.model", self.tgt_proto)]:
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(proto)
        state = {"class": type(self).__name__, "src_vocab": self.src_vocab, "tgt_vocab": self.tgt_vocab,
                 "specials": self.specials, "langs": self.langs}
        with open(os.path.join(out_dir, "codec.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    @classmethod
    def load(cls, out_dir):
        with open(os.path.join(out_dir, "codec.json"), encoding="utf-8") as f:
            s = json.load(f)
        protos = {}
        for name in ["spm_full", "spm_src", "spm_tgt"]:
            with open(os.path.join(out_dir, f"{name}.model"), "rb") as f:
                protos[name] = f.read()
        return cls(protos["spm_full"], s["src_vocab"], s["tgt_vocab"], s["specials"], s["langs"],
                   protos["spm_src"], protos["spm_tgt"])


class BhashaverseCodec(SpmCodec):
    # Language is selected by a text tag "###src-to-tgt### ", ids come from the fairseq dictionary, decoding starts from EOS

    def encode_src(self, text, src, tgt):
        tagged = f"###{self.flores(src)}-to-{self.flores(tgt)}### {text}"
        return self.src_ids(self.sp_src.encode(tagged, out_type=str)) + [self.eos]

    def encode_tgt(self, text, tgt):
        return self.tgt_ids(self.sp_tgt.encode(text, out_type=str)) + [self.eos]

    def decode(self, ids):
        drop = {self.specials["bos"], self.pad, self.eos}
        pieces = [self.tgt_id2tok.get(i, "<unk>") for i in ids if i not in drop]
        return self.sp_full.decode(pieces).strip()

    def gen_kwargs(self, tgt):
        return {"decoder_start_token_id": self.eos}

    def save(self, out_dir):
        super().save(out_dir)
        fairseq = {"src": self.src_vocab, "tgt": self.tgt_vocab, "special": self.specials}
        with open(os.path.join(out_dir, "fairseq_dict.json"), "w", encoding="utf-8") as f:
            json.dump(fairseq, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "onemtv3b_spm.model"), "wb") as f:
            f.write(self.src_proto)


class NllbCodec(SpmCodec):
    # Source: [src_lang] pieces </s>; target: [tgt_lang] pieces </s>; generation starts from </s> and forces the target language token

    def lang_id(self, lang, side):
        vocab = self.src_vocab if side == "src" else self.tgt_vocab
        return vocab[self.flores(lang)]

    def encode_src(self, text, src, tgt):
        return [self.lang_id(src, "src")] + self.src_ids(self.sp_src.encode(text, out_type=str)) + [self.eos]

    def encode_tgt(self, text, tgt):
        return [self.lang_id(tgt, "tgt")] + self.tgt_ids(self.sp_tgt.encode(text, out_type=str)) + [self.eos]

    def decode(self, ids):
        lang_tokens = {v["flores"] for v in self.langs.values()}
        drop = set(self.specials.values()) - {self.unk}
        pieces = [self.tgt_id2tok[i] for i in ids if i not in drop and i in self.tgt_id2tok]
        return self.sp_full.decode([p for p in pieces if p not in lang_tokens]).strip()

    def gen_kwargs(self, tgt):
        return {"decoder_start_token_id": self.eos, "forced_bos_token_id": self.lang_id(tgt, "tgt")}


CODECS = {c.__name__: c for c in [BhashaverseCodec, NllbCodec]}


def load_codec(out_dir):
    with open(os.path.join(out_dir, "codec.json"), encoding="utf-8") as f:
        name = json.load(f)["class"]
    return CODECS[name].load(out_dir)
