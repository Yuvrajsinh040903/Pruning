# Static vocabulary pruning for encoder-decoder MT (Bhashaverse, NLLB)

## Setup

```
pip install "transformers==4.45.2" torch sentencepiece protobuf safetensors sacrebleu datasets huggingface_hub
```

Bhashaverse pins transformers 4.45.2. The code also runs on transformers 5.x.

## Data

One TSV per language pair with a header row containing `source` and `target` (names set by `SRC_COLUMN` /
`TGT_COLUMN`), located by `DATA_TEMPLATE` in `config.py`:

```
data/train/eng-hin.tsv  data/train/eng-tam.tsv  ...   <- SELECT_SPLIT: tokens are chosen from here
data/test/eng-hin.tsv   data/test/eng-tam.tsv   ...   <- EVAL_SPLIT: BLEU / chrF++ / fidelity
```

Columns are split on tabs only; quotes inside sentences are kept as text. Extra columns are ignored.

`python prepare_flores.py` writes FLORES+ devtest to `data/test` and dev to `data/dev`.
Never select tokens from the test split.

## Run

Edit `config.py`, then `python run.py`.

Finished experiments are skipped on re-run. Token counts, model-generated tokens and baseline
translations are cached in `OUTPUT_DIR/cache/`; delete that folder after changing data or generation settings.

## Outputs

- `OUTPUT_DIR/results.csv`: one row per (model, selection setting, embed mode, target group, target language)
- `OUTPUT_DIR/models/<model>/<setting>_<mode>_<group>/`: pruned weights, configs, hypotheses, `pruning_meta.json`
  and the pruned tokenizer. For Bhashaverse this includes a pruned `fairseq_dict.json` + `onemtv3b_spm.model`
  that drop straight into the original `hf_inference.py`.

Reload a pruned model with:

```python
import torch
from pruning import load_pruned
model, codec = load_pruned("outputs/vocab_pruning/models/nllb/seen_union_hin")
ids = codec.encode_src("Hello world", "eng", "hin")
out = model.generate(input_ids=torch.tensor([ids]), **codec.gen_kwargs("hin"), num_beams=5)
print(codec.decode(out[0].tolist()))
```

## Result columns

| column | meaning |
|---|---|
| vocab_src / vocab_tgt | rows kept in the encoder embedding / decoder embedding + lm_head |
| params_M, weights_MB, alloc_MB, peak_MB | parameter count, weight bytes, GPU memory after load, peak GPU memory during generation |
| bleu, chrf++ (_base / _pruned) | quality before and after pruning |
| identical_outputs | fraction of test sentences translated identically |
| src_oov_rate, ref_oov_rate | test tokens outside the kept set (source / reference) |
| src_sent_changed | test sources whose tokenization changes after pruning |
| logit_max_abs_diff, argmax_agreement | kept-token logits of pruned vs original on fully in-vocabulary sentences (should be ~0 and 1.0) |
| removed_mass_mean / max | probability the original model puts on removed tokens |
| argmax_removed_rate | positions where the original model's top token was removed |

## Adding a model

Write a loader in `models.py` returning `(model, codec)` and register it in `LOADERS`. If the tokenizer is not
SentencePiece-based, subclass `SpmCodec` in `tokenization.py`. If embedding modules live elsewhere, edit `EMBEDDING_PATHS` in `pruning.py`.
