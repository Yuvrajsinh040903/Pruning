import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_ROOT = Path("/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/IN22_Gen/pruning")
SPLIT = "test"              # "test" = FLORES devtest, "dev" = FLORES dev (as saved by the download script)
SRC_COLUMN = "source"
TGT_COLUMN = "target"

MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
# OUT_DIR = Path(MODEL_PATH) / "in22_gen_inference"
OUT_DIR = Path("/home/extra_storage_dipti/yuvrajsinh.bodana/Pruning_Experiments/Base/Qwen_3_4B_INST/in22_gen_inference")
PAIRS = None                # None = every eng-*.tsv in the split folder; or e.g. ["eng-hin", "eng-tam"]
PROMPT_STYLE = "chat"       # "chat" (Qwen/Llama-style) or "tgemma" (TranslateGemma template)
SKIP_EXISTING = True        # don't redo pairs whose output TSV already exists
MAX_NEW_TOKENS = 512
GPU_MEM_UTIL = 0.9
TP_SIZE = 1

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

# ----------------------------------------------------------------------------
# Language info (keys = the 3-letter codes used in the file names)
# ----------------------------------------------------------------------------
LANGS = {
    "eng": {"name": "English",   "iso": "en", "flores": "eng_Latn"},
    "hin": {"name": "Hindi",     "iso": "hi", "flores": "hin_Deva"},
    "tel": {"name": "Telugu",    "iso": "te", "flores": "tel_Telu"},
    "guj": {"name": "Gujarati",  "iso": "gu", "flores": "guj_Gujr"},
    "tam": {"name": "Tamil",     "iso": "ta", "flores": "tam_Taml"},
    "ben": {"name": "Bengali",   "iso": "bn", "flores": "ben_Beng"},
    "urd": {"name": "Urdu",      "iso": "ur", "flores": "urd_Arab"},
    "kan": {"name": "Kannada",   "iso": "kn", "flores": "kan_Knda"},
}


# ----------------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------------
def build_chat_messages(src_lang, tgt_lang, src):
    s, t = LANGS[src_lang]["name"], LANGS[tgt_lang]["name"]
    prompt = (f"Translate following sentence from {s} to {t}.\n{src}\n"
              f"RESPONSE FORMAT:<ONLY_TRANSLATED_TEXT>")
    return [{"role": "user", "content": prompt}]


def build_tgemma_prompt(tokenizer, src_lang, tgt_lang, src):
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": LANGS[src_lang]["iso"],
            "target_lang_code": LANGS[tgt_lang]["iso"],
            "text": src,
        }],
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_pair_tsv(path):
    # The download script writes raw lines without quoting, and FLORES text contains
    # quote characters, so QUOTE_NONE is required for a correct parse.
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False,
                     dtype=str, encoding="utf-8")
    missing = {SRC_COLUMN, TGT_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}, found {list(df.columns)}")
    return df[[SRC_COLUMN, TGT_COLUMN]]


def discover_pairs():
    split_dir = DATA_ROOT
    files = sorted(split_dir.glob("*-*.tsv"))
    if not files:
        raise FileNotFoundError(f"No pair TSVs found in {split_dir}")
    found = {}
    for f in files:
        pair = f.stem                       # "eng-hin"
        src_lang, tgt_lang = pair.split("-")
        if src_lang not in LANGS or tgt_lang not in LANGS:
            print(f"|> skipping {f.name}: language not in LANGS")
            continue
        if PAIRS is None or pair in PAIRS:
            found[pair] = (src_lang, tgt_lang, f)
    return found


def score_pair(hyps, refs):
    try:
        import sacrebleu
    except ImportError:
        return None
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score    # chrF++
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    return chrf, bleu


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs()
    todo = {p: v for p, v in pairs.items()
            if not (SKIP_EXISTING and (OUT_DIR / f"{p}.tsv").exists())}
    print(f"|> split={DATA_ROOT}: {len(pairs)} pairs found, {len(todo)} to run: {list(todo)}")
    if not todo:
        print("|> Nothing to do.")
        return

    # Load every pending pair into one frame, tagged by pair
    frames = []
    for pair, (src_lang, tgt_lang, path) in todo.items():
        df = read_pair_tsv(path)
        df["pair"], df["src_lang"], df["tgt_lang"] = pair, src_lang, tgt_lang
        frames.append(df)
        print(f"|> loaded {path.name}: {len(df)} sentences")
    run_df = pd.concat(frames, ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        dtype=torch.bfloat16,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=True,
        tensor_parallel_size=TP_SIZE,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 0},
    )
    sampling = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0,
                              stop=["</s>", "<|im_end|>", "<|endoftext|>"])

    # One vLLM batch for all pairs (outputs come back in input order)
    rows = list(run_df.itertuples(index=False))
    if PROMPT_STYLE == "chat":
        convs = [build_chat_messages(r.src_lang, r.tgt_lang, getattr(r, SRC_COLUMN)) for r in rows]
        outputs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    else:
        prompts = [build_tgemma_prompt(tokenizer, r.src_lang, r.tgt_lang, getattr(r, SRC_COLUMN))
                   for r in rows]
        outputs = llm.generate(prompts, sampling)

    # Same cleaning as the download script, so mt and references are formatted alike
    run_df["mt"] = [" ".join(o.outputs[0].text.replace("\t", " ").split()) for o in outputs]

    summary = []
    for pair, g in run_df.groupby("pair", sort=True):
        out_file = OUT_DIR / f"{pair}.tsv"
        g[[SRC_COLUMN, TGT_COLUMN, "mt"]].to_csv(out_file, sep="\t", index=False,
                                                 quoting=csv.QUOTE_NONE, escapechar="\\")
        row = {"pair": pair, "n": len(g)}
        scores = score_pair(g["mt"].tolist(), g[TGT_COLUMN].tolist())
        if scores:
            row["chrF++"], row["BLEU"] = round(scores[0], 2), round(scores[1], 2)
        summary.append(row)
        print(f"|> saved {out_file}"
              + (f"  chrF++={row['chrF++']}  BLEU={row['BLEU']}" if scores else ""))

    # Merge with an existing summary so previously finished pairs are kept
    summ_file = OUT_DIR / "_summary.tsv"
    summ = pd.DataFrame(summary)
    if summ_file.exists():
        old = pd.read_csv(summ_file, sep="\t")
        summ = pd.concat([old[~old["pair"].isin(summ["pair"])], summ]).sort_values("pair")
    summ.to_csv(summ_file, sep="\t", index=False)
    print(f"|> summary written to {summ_file}")


if __name__ == "__main__":
    main()
