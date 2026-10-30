import torch

OUTPUT_DIR = "outputs_bpcc_wo_hin/vocab_pruning"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# fp32 keeps the logit-equivalence check exact; switch to torch.bfloat16 for speed once the check passes
DTYPE = torch.float32
SEED = 0

MODELS = ["bhashaverse"]#, "nllb"]
MODEL_PATHS = {
    "bhashaverse": "ltrciiith/bhashaverse",
    # "nllb": "facebook/nllb-200-distilled-600M",
}

SRC_LANG = "eng"
TGT_LANGS = ["hin","tam", "tel", "ben", "kan"]

# Every group becomes one pruned model: single-language groups give one model per pair, "all" gives one model for all pairs
TARGET_GROUPS = {
    # "hin": ["hin"],
    # "tam": ["tam"],
    # "tel": ["tel"],
    # "ben": ["ben"],
    # "kan": ["kan"],
    "all": TGT_LANGS,
}

LANGS = {
    "eng": {"flores": "eng_Latn", "scripts": ["latin"]},
    # "hin": {"flores": "hin_Deva", "scripts": ["devanagari"]},
    "tam": {"flores": "tam_Taml", "scripts": ["tamil"]},
    "tel": {"flores": "tel_Telu", "scripts": ["telugu"]},
    "ben": {"flores": "ben_Beng", "scripts": ["bengali"]},
    "kan": {"flores": "kan_Knda", "scripts": ["kannada"]},
}

SCRIPT_RANGES = {
    "common": [(0x0020, 0x007E), (0x00A0, 0x00BF), (0x2000, 0x206F), (0x20A0, 0x20CF), (0x0964, 0x0965)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
}

# One TSV per language pair with a header row; {split} is the folder, {tgt} the target language code
DATA_TEMPLATE = "bpcc_data/{split}/eng-{tgt}.tsv"
SRC_COLUMN = "source"
TGT_COLUMN = "target"
SELECT_SPLIT = "train"
EVAL_SPLIT = "test"
MAX_SELECT_LINES = None
MAX_EVAL_LINES = None

# "union": encoder, decoder and lm_head keep one shared set (source + target tokens)
# "split": encoder keeps source tokens only, decoder and lm_head keep target tokens only (unties shared matrices)
EMBED_MODES = ["union", "split"]

# min_count / top_k / coverage are applied per language on corpus counts
# char_fallback keeps every single-character piece of the relevant scripts so unseen words never become <unk>
# model_outputs adds every token the original model generates on the selection data (VocabTrim-style)
SELECTION_SETTINGS = [
    {"name": "seen", "min_count": 1, "top_k": None, "coverage": None, "char_fallback": True, "model_outputs": False},
    {"name": "seen_gen", "min_count": 1, "top_k": None, "coverage": None, "char_fallback": True, "model_outputs": True},
    # {"name": "cov99.9", "min_count": 1, "top_k": None, "coverage": 0.999, "char_fallback": True, "model_outputs": False},
    # {"name": "top8k", "min_count": 1, "top_k": 8000, "coverage": None, "char_fallback": True, "model_outputs": False},
]
MODEL_OUTPUT_LINES = 2000

GEN_KWARGS = {
    "bhashaverse": dict(num_beams=5, max_new_tokens=256, no_repeat_ngram_size=3, repetition_penalty=1.3, early_stopping=True),
    # "nllb": dict(num_beams=5, max_new_tokens=256),
}
GEN_BATCH_SIZE = 32
TF_BATCH_SIZE = 16

# "13a" for BLEU; "flores200" gives spBLEU (downloads the FLORES-200 spm model on first use)
BLEU_TOKENIZE = "13a"
FIDELITY_SENTENCES = 64

SAVE_MODELS = True
