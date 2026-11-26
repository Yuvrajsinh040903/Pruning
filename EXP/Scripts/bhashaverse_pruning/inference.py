import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import time

import torch

from bhashaverse import Codec, load_model

# name -> saved model folder (or HF repo); outputs go to OUTPUT_ROOT/<name>/<dataset>_inference/<src>_<tgt>.tsv
MODELS = {
    "teacher": "ltrciiith/bhashaverse",
    "distilled": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Scripts/bhashaverse_pruning/outputs/distil_bhashaverse/distilbert_enc9_dec9/best",
}
OUTPUT_ROOT = "outputs/inference"

# dataset name -> folder with one TSV per pair (header columns source/target), named by INPUT_TEMPLATE
DATASETS = {
    "flores_pruning": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/FLORES/pruning",
    "flores_extra": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/FLORES/extra_data",
    "in22_conv_pruning": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/IN22_Conv/pruning",
    "in22_conv_extra": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/IN22_Conv/extra_data",
    "in22_gen_pruning": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/IN22_Gen/pruning",
    "in22_gen_extra": "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/IN22_Gen/extra_data"
}
INPUT_TEMPLATE = "{src}-{tgt}.tsv"
SRC_COLUMN, TGT_COLUMN = "source", "target"
#PAIRS = [("eng", "hin"), ("eng", "tam"), ("eng", "tel"), ("eng", "ben"), ("eng", "kan")]
PAIRS = [
    ("eng", "asm"),
    ("eng", "mal"),
    ("eng", "mar"),
    ("eng", "npi"),
    ("eng", "ory"),
    ("eng", "pan"),
    ("eng", "san"),
    ("eng", "ben"),
    ("eng", "guj"),
    ("eng", "hin"),
    ("eng", "kan"),
    ("eng", "tam"),
    ("eng", "tel"),
    ("eng", "urd"),
]
MAX_SENTENCES = None
OVERWRITE = False

FLORES_CODES = {"eng": "eng_Latn", "hin": "hin_Deva", "tam": "tam_Taml", "tel": "tel_Telu", "ben": "ben_Beng",
                "kan": "kan_Knda", "mar": "mar_Deva", "guj": "guj_Gujr", "mal": "mal_Mlym", "pan": "pan_Guru",
                "ory": "ory_Orya", "urd": "urd_Arab", "asm": "asm_Beng", "npi": "npi_Deva", "san": "san_Deva"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 32
MAX_SRC_LEN = 256
# Same generation settings as hf_inference.py
GEN_KWARGS = dict(num_beams=5, max_new_tokens=256, early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=1.3)


def clean(text):
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        si, ti = header.index(SRC_COLUMN), header.index(TGT_COLUMN)
        sources, refs = [], []
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(si, ti) or not cols[si].strip():
                continue
            sources.append(clean(cols[si]))
            refs.append(clean(cols[ti]))
            if MAX_SENTENCES and len(sources) >= MAX_SENTENCES:
                break
    return sources, refs


def write_tsv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(clean(str(x)) for x in row) + "\n")
    os.replace(tmp, path)


@torch.no_grad()
def translate(model, codec, sources, src, tgt):
    enc = [codec.encode_src(s, src, tgt) for s in sources]
    # Longest first so an out-of-memory error shows up on the first batch
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]), reverse=True)
    out = [None] * len(enc)
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        ids = torch.full((len(batch), max(len(enc[i]) for i in batch)), codec.pad, dtype=torch.long)
        for row, i in enumerate(batch):
            ids[row, :len(enc[i])] = torch.tensor(enc[i])
        ids = ids.to(DEVICE)
        generated = model.generate(input_ids=ids, attention_mask=(ids != codec.pad).long(),
                                   decoder_start_token_id=codec.eos, **GEN_KWARGS)
        for i, g in zip(batch, generated.tolist()):
            out[i] = codec.decode(g)
    return out


def main():
    for name, path in MODELS.items():
        print(f"== {name}: {path}")
        model = load_model(path, DTYPE).to(DEVICE)
        codec = Codec(path, FLORES_CODES, MAX_SRC_LEN, 10 ** 6)
        for dataset, data_dir in DATASETS.items():
            out_dir = os.path.join(OUTPUT_ROOT, name, f"{dataset}_inference")
            os.makedirs(out_dir, exist_ok=True)
            for src, tgt in PAIRS:
                in_path = os.path.join(data_dir, INPUT_TEMPLATE.format(src=src, tgt=tgt))
                out_path = os.path.join(out_dir, f"{src}_{tgt}.tsv")
                if not os.path.exists(in_path):
                    print(f"  skip {dataset} {src}_{tgt}: {in_path} not found")
                    continue
                if os.path.exists(out_path) and not OVERWRITE:
                    print(f"  skip {dataset} {src}_{tgt}: already done")
                    continue
                sources, refs = read_pairs(in_path)
                start = time.perf_counter()
                hyps = translate(model, codec, sources, src, tgt)
                write_tsv(out_path, ["source", "target", "hypothesis"], zip(sources, refs, hyps))
                print(f"  {dataset} {src}_{tgt}: {len(hyps)} sentences in {time.perf_counter() - start:.0f}s -> {out_path}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
