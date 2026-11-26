import torch

TEACHER_PATH = "ltrciiith/bhashaverse"
OUTPUT_DIR = "outputs/distil_bhashaverse"
# None -> "<LAYER_RULE>_enc<kept>_dec<kept>"
RUN_NAME = None

SRC_LANG = "eng"
TGT_LANGS = ["hin", "tam", "tel", "ben", "kan", "guj", "urd"]
FLORES_CODES = {"eng": "eng_Latn", "hin": "hin_Deva", "tam": "tam_Taml", "tel": "tel_Telu",
                "ben": "ben_Beng", "kan": "kan_Knda", "mar": "mar_Deva", "guj": "guj_Gujr",
                "mal": "mal_Mlym", "pan": "pan_Guru", "ory": "ory_Orya", "urd": "urd_Arab", "asm": "asm_Beng"}

# One TSV per pair with header columns source/target; if the dev file is missing, the last DEV_SIZE train pairs are held out
DATA_TEMPLATE = "/home/priyanka/Pruning/EXP/Data/bpcc_data/{split}/eng-{tgt}.tsv"
SRC_COLUMN = "source"
TGT_COLUMN = "target"
TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT = "train", "dev", "test"
MAX_TRAIN_PAIRS_PER_LANG = 200_000
DEV_SIZE = 200
MAX_TEST_PAIRS = None
MAX_SRC_LEN = 256
MAX_TGT_LEN = 256

# Layer selection
# "distilbert": keep n//2 layers, even layers from the bottom and odd-offset layers from the top, so first and last are kept
#               (12 layers -> [0, 2, 4, 7, 9, 11], exactly DistilBERT; 18 layers -> [0, 2, 4, 6, 8, 11, 13, 15, 17])
# "alternate":  every other layer from 0, plus the last one (18 layers -> 10 kept)
# "custom":     the lists below
LAYER_RULE = "distilbert"
PRUNE_ENCODER = True
PRUNE_DECODER = True
CUSTOM_ENCODER_LAYERS = None
CUSTOM_DECODER_LAYERS = None

# DistilBERT objective: ALPHA_KD * soft-target KL + ALPHA_CE * reference cross-entropy + ALPHA_COS * hidden-state cosine
TEMPERATURE = 2.0
ALPHA_KD = 5.0
ALPHA_CE = 2.0
ALPHA_COS = 1.0
# "last": final encoder and decoder states (DistilBERT); "all": every student layer vs the teacher layer it was copied from
COS_LAYERS = "last"
COS_STACKS = ["encoder", "decoder"]

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16
TEACHER_DTYPE = torch.bfloat16
STUDENT_DROPOUT = 0.1
FREEZE_EMBEDDINGS = False
GRADIENT_CHECKPOINTING = False
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
MAX_STEPS = 20_000
MAX_TOKENS_PER_BATCH = 4096
GRAD_ACCUM = 4
MAX_GRAD_NORM = 1.0
LOG_EVERY = 50
EVAL_EVERY = 1000
# Dev loss used to keep the best checkpoint: "ce" (reference NLL), "kd" or "total"
SELECT_BEST_BY = "ce"
SEED = 42

# Generation, same settings as hf_inference.py
BEAM_SIZE = 5
MAX_NEW_TOKENS = 256
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.3
EVAL_BATCH_SIZE = 32
EVAL_DTYPE = torch.bfloat16
BLEU_TOKENIZE = "13a"

EVALUATE_TEACHER = True
EVALUATE_BEFORE_DISTILL = True
