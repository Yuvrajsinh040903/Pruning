import csv
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict

import sentencepiece as spm
from huggingface_hub import hf_hub_download

# Standalone: nothing is read from config.py, so changing the experiment config does not affect this analysis

# Hugging Face repo id or a local folder holding fairseq_dict.json and onemtv3b_spm.model
MODEL_PATH = "ltrciiith/bhashaverse"

# One entry per pruning run to compare; each folder holds the experiment folders written by run.py
RUNS = {
    "flores": "outputs/vocab_pruning/models/bhashaverse",
    "bpcc": "outputs_bpcc/vocab_pruning/models/bhashaverse",
}
OUT_DIR = "outputs/vocab_analysis"

SRC_LANG = "eng"
TGT_LANGS = ["hin", "tam", "tel", "ben", "kan"]
FLORES_CODES = {"eng": "eng_Latn", "hin": "hin_Deva", "tam": "tam_Taml", "tel": "tel_Telu",
                "ben": "ben_Beng", "kan": "kan_Knda"}

# Test set used for the coverage table: one TSV per pair with a header row
TEST_TEMPLATE = "data/test/eng-{tgt}.tsv"
SRC_COLUMN = "source"
TGT_COLUMN = "target"
MAX_TEST_LINES = None

# Scripts printed as columns; any other script still goes to the TSVs
SHOW_SCRIPTS = ["latin", "devanagari", "bengali", "tamil", "telugu", "kannada", "common"]

SPECIAL_PREFIXES = ("<", "madeupword")
# Scripts used by Bhashaverse's 36 languages: Latin, Devanagari, Bengali, Gujarati, Gurmukhi, Odia, Tamil, Telugu,
# Kannada, Malayalam, Sinhala, Arabic, Ol Chiki, Warang Citi, Meetei Mayek. Anything else is reported under its own name.
NAME_TO_SCRIPT = {"OL CHIKI": "ol_chiki", "MEETEI MAYEK": "meetei_mayek", "WARANG CITI": "warang_citi",
                  "ORIYA": "odia", "VEDIC": "devanagari", "CJK": "cjk"}
# Unicode names starting with these words carry no script (digits, superscripts, combining accents, ...)
NO_SCRIPT_WORDS = {"DIGIT", "SUPERSCRIPT", "SUBSCRIPT", "VULGAR", "ROMAN", "CIRCLED", "PARENTHESIZED",
                   "FULLWIDTH", "COMBINING", "MODIFIER"}


def char_script(ch):
    # The script comes from the character's Unicode name, e.g. "DEVANAGARI LETTER KA" -> devanagari
    cat = unicodedata.category(ch)
    # Punctuation (including the danda), symbols, spaces and format chars such as ZWJ/ZWNJ are shared by all languages
    if cat[0] in "PSZC":
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return "unknown"
    for prefix, script in NAME_TO_SCRIPT.items():
        if name.startswith(prefix):
            return script
    first = name.split(" ")[0]
    return None if first in NO_SCRIPT_WORDS else first.lower()


def token_script(token):
    if token.startswith(SPECIAL_PREFIXES):
        return "special"
    core = token.replace("▁", "")
    if not core:
        return "common"
    scripts = {s for s in map(char_script, core) if s}
    if not scripts:
        return "common"
    return scripts.pop() if len(scripts) == 1 else "mixed"


def resolve(filename):
    local = os.path.join(MODEL_PATH, filename)
    return local if os.path.isdir(MODEL_PATH) else hf_hub_download(MODEL_PATH, filename)


class Codec:
    # Base Bhashaverse tokenization: "###src-to-tgt### " tag + SentencePiece pieces mapped through the fairseq dictionary
    def __init__(self):
        with open(resolve("fairseq_dict.json"), encoding="utf-8") as f:
            fs = json.load(f)
        self.src_vocab, self.tgt_vocab = fs["src"], fs["tgt"]
        self.unk, self.eos = fs["special"]["unk"], fs["special"]["eos"]
        self.sp = spm.SentencePieceProcessor(model_file=resolve("onemtv3b_spm.model"))

    def encode_src(self, text, src, tgt):
        tagged = f"###{FLORES_CODES[src]}-to-{FLORES_CODES[tgt]}### {text}"
        return [self.src_vocab.get(p, self.unk) for p in self.sp.encode(tagged, out_type=str)] + [self.eos]

    def encode_tgt(self, text, tgt):
        return [self.tgt_vocab.get(p, self.unk) for p in self.sp.encode(text, out_type=str)] + [self.eos]


def load_test_pairs(tgt):
    path = TEST_TEMPLATE.format(tgt=tgt)
    csv.field_size_limit(sys.maxsize)
    sources, refs = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row[SRC_COLUMN] is None or row[TGT_COLUMN] is None:
                continue
            sources.append(row[SRC_COLUMN].strip())
            refs.append(row[TGT_COLUMN].strip())
            if MAX_TEST_LINES and len(sources) >= MAX_TEST_LINES:
                break
    return sources, refs


def id_scripts(vocab):
    return {i: token_script(tok) for tok, i in vocab.items()}


def experiments():
    for run, root in RUNS.items():
        if not os.path.isdir(root):
            print(f"skipping {run}: {root} not found")
            continue
        for exp in sorted(os.listdir(root)):
            meta_path = os.path.join(root, exp, "pruning_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                yield run, exp, set(meta["keep_src"]), set(meta["keep_tgt"])


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


def print_table(title, rows, label_keys, value_key, scripts):
    print(f"\n{title}")
    groups = defaultdict(dict)
    for r in rows:
        groups[tuple(r[k] for k in label_keys)][r["script"]] = r[value_key]
    width = max(len(" | ".join(k)) for k in groups) + 2
    print(" " * width + "".join(f"{s:>12}" for s in scripts))
    for key, vals in groups.items():
        cells = "".join(f"{vals[s]:>12.1f}" if isinstance(vals.get(s), float) else f"{vals.get(s, '-'):>12}" for s in scripts)
        print(f"{' | '.join(key):<{width}}{cells}")


def vocab_distribution(codec):
    rows = []
    for side, vocab in [("encoder", codec.src_vocab), ("decoder", codec.tgt_vocab)]:
        counts = Counter(id_scripts(vocab).values())
        total = sum(counts.values())
        for script, n in counts.most_common():
            rows.append({"side": side, "script": script, "tokens": n, "pct_of_vocab": 100 * n / total})
    return rows


def kept_by_script(codec):
    scripts = {"encoder": id_scripts(codec.src_vocab), "decoder": id_scripts(codec.tgt_vocab)}
    base = {side: Counter(s.values()) for side, s in scripts.items()}
    rows = []
    for run, exp, keep_src, keep_tgt in experiments():
        for side, keep in [("encoder", keep_src), ("decoder", keep_tgt)]:
            kept = Counter(scripts[side][i] for i in keep if i in scripts[side])
            for script, n_base in base[side].most_common():
                rows.append({"run": run, "experiment": exp, "side": side, "script": script, "base_tokens": n_base,
                             "kept_tokens": kept[script], "kept_pct": 100 * kept[script] / n_base})
    return rows


def test_coverage(codec):
    # Share of test token occurrences, per script, that are still in the kept vocabulary
    scripts = {"source": id_scripts(codec.src_vocab), "reference": id_scripts(codec.tgt_vocab)}
    occurrences = {}
    for tgt in TGT_LANGS:
        sources, refs = load_test_pairs(tgt)
        occurrences[tgt] = {"source": Counter(i for s in sources for i in codec.encode_src(s, SRC_LANG, tgt)),
                            "reference": Counter(i for r in refs for i in codec.encode_tgt(r, tgt))}
    rows = []
    for run, exp, keep_src, keep_tgt in experiments():
        for tgt, by_side in occurrences.items():
            for side, keep in [("source", keep_src), ("reference", keep_tgt)]:
                total, covered = Counter(), Counter()
                for i, n in by_side[side].items():
                    script = scripts[side].get(i, "unknown")
                    total[script] += n
                    covered[script] += n if i in keep else 0
                for script, n in total.most_common():
                    rows.append({"run": run, "experiment": exp, "tgt": tgt, "side": side, "script": script,
                                 "occurrences": n, "covered_pct": 100 * covered[script] / n})
    return rows


def main():
    codec = Codec()

    dist = vocab_distribution(codec)
    write_tsv(os.path.join(OUT_DIR, "vocab_by_script.tsv"), dist)
    print("\nBase vocabulary by script (decoder side; encoder side is in the TSV)")
    for r in dist:
        if r["side"] == "decoder":
            print(f"  {r['script']:<14}{r['tokens']:>8}  {r['pct_of_vocab']:5.1f}%")

    kept = kept_by_script(codec)
    if not kept:
        print("\nNo pruned experiments found; check RUNS")
        return
    write_tsv(os.path.join(OUT_DIR, "kept_by_script.tsv"), kept)
    for side in ["encoder", "decoder"]:
        print_table(f"% of base {side} tokens kept, per script", [r for r in kept if r["side"] == side],
                    ["run", "experiment"], "kept_pct", SHOW_SCRIPTS)

    cov = test_coverage(codec)
    write_tsv(os.path.join(OUT_DIR, "test_coverage_by_script.tsv"), cov)
    for side in ["source", "reference"]:
        print_table(f"% of test {side} token occurrences still in the vocabulary, per script",
                    [r for r in cov if r["side"] == side], ["run", "experiment", "tgt"], "covered_pct", SHOW_SCRIPTS)
    print(f"\nTSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()