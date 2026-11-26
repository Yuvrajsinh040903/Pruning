import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
import json
import random
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llama_vocab import (
    BPE, budget, budget_table, build_corpus, check_simulation, compare_logits, coverage_report, kept_by_script,
    logits_before, prune_model, prune_tokenizer, read_pairs, render, select_vocab, tokenization_changes,
)

base_model = "meta-llama/Llama-3.2-3B-Instruct"
model_path = "/home/extra_storage_dipti/yuvrajsinh.bodana/Pruning_Experiments/EXP_cosine_kd_base_mse_ce_Sun_Nov_15_04-16-48_2026_high_lr_1e-3/Pruned_Model_28_to_20_cosine_mse_ce"
out_dir = Path(model_path).parent / "vocab_pruned"
#out_dir = "/home/extra_storage_dipti/yuvrajsinh.bodana/Pruning_Experiments/Base/"

langs = {"hin": "Hindi", "tel": "Telugu", "guj": "Gujarati", "tam": "Tamil", "ben": "Bengali", "urd": "Urdu", "kan": "Kannada"}
select_template = "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/bpcc_data/train/eng-{tgt}.tsv"
report_template = "/home/yuvrajsinh.bodana/pruning_git/Pruning/EXP/Data/flores_data/{split}/eng-{tgt}.tsv"
report_splits = ["dev", "test"]
src_col, tgt_col = "source", "target"
max_lines_per_lang = None
min_count = 1
keep_digits = True
extra_texts = []

select_only = False
check_sentences = 16
check_max_len = 384
sim_check_texts = 2000
generate_examples = True
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0

random.seed(seed)
tok = AutoTokenizer.from_pretrained(model_path)
corpus = build_corpus(tok, langs, select_template, src_col, tgt_col, max_lines_per_lang)
print({k: len(v) for k, v in corpus.items()})

kept_ids, bpe, stats = select_vocab(tok, corpus, min_count, keep_digits, extra_texts)
pool = [t for texts in corpus.values() for t in texts]
sample = random.sample(pool, min(sim_check_texts, len(pool)))
check_simulation(tok, bpe, sample)
print(f"kept {len(kept_ids):,} of {len(tok):,} tokens", stats)

cfg = AutoConfig.from_pretrained(model_path)
info = budget(base_model, cfg, len(kept_ids))
print(json.dumps(info, indent=2))
table = budget_table(info, info["base_layers"] // 2)
print(table.round(2).to_string(index=False))
if select_only:
    raise SystemExit

out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "budget_table.tsv", sep="\t", index=False)
remap = prune_tokenizer(tok, kept_ids, out)
new_tok = AutoTokenizer.from_pretrained(out)
assert len(new_tok) == len(kept_ids), (len(new_tok), len(kept_ids))
tok_check = tokenization_changes(tok, new_tok, kept_ids, sample)
print("selection texts:", tok_check)

model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
check_texts = random.sample(pool, min(check_sentences, len(pool)))
before = logits_before(model, tok, check_texts, kept_ids, check_max_len)
tied = prune_model(model, kept_ids, remap)
logit_check = compare_logits(model, new_tok, check_texts, before, check_max_len)
print("logits:", logit_check)
assert info["params_final"] == sum(p.numel() for p in model.parameters())

if generate_examples:
    for code, name in langs.items():
        path = Path(report_template.format(split="dev", tgt=code))
        if not path.exists():
            continue
        src = read_pairs(path, src_col, tgt_col, 1)[0][0]
        ids = new_tok(render(new_tok, name, src), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        gen = model.generate(ids, max_new_tokens=96, do_sample=False)
        print(f"[{code}] {new_tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)}")

model.save_pretrained(out)
coverage = coverage_report(tok, new_tok, kept_ids, langs, report_template, src_col, tgt_col, report_splits)
coverage.to_csv(out / "coverage.tsv", sep="\t", index=False)
print(coverage.round(4).to_string(index=False))
kept_by_script(tok, kept_ids).to_csv(out / "kept_by_script.tsv", sep="\t", index=False)
pd.DataFrame({"new_id": range(len(kept_ids)), "old_id": kept_ids, "token": tok.convert_ids_to_tokens(kept_ids)}).to_csv(
    out / "kept_tokens.tsv", sep="\t", index=False
)
meta = {"source_model": model_path, "base_model": base_model, "selection_data": select_template, "langs": langs,
        "min_count": min_count, "tied_embeddings": tied, "selection": stats, "budget": info,
        "selection_tokenization": tok_check, "logit_check": logit_check}
(out / "vocab_pruning_info.json").write_text(json.dumps(meta, indent=2))
print(f"Saved to {out}")
