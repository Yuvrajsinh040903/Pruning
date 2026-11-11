import itertools
import math
import time
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments
from pruning_core import PROMPT, read_tsv

TERMS = ("mse", "kl", "ce")


@dataclass
class RecoveryConfig:
    mse_weight: float = 1.0
    kl_weight: float = 0.0
    ce_weight: float = 0.0
    balance_terms: bool = True
    scale_batches: int = 16
    temperature: float = 1.0
    normalize_mse: bool = False
    kl_on_response_only: bool = True
    lr: float = 1e-5
    warmup_frac: float = 0.05
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 8
    eval_steps: int = 50
    patience: int = 3
    max_len: int = 512
    train_per_lang: int = 4000
    eval_per_lang: int = 50
    seed: int = 0

    def weight(self, term):
        return getattr(self, f"{term}_weight")

    @property
    def needs_logits(self):
        return self.kl_weight > 0 or self.ce_weight > 0


@dataclass
class Target:
    trainable_idx: int
    trainable_orig: int
    base_idx: int


def make_target(active_ids, removed_idx, num_base_layers):
    # trainable layer must reproduce the base model's input to the next surviving layer
    t = max(removed_idx - 1, 0)
    nxt = active_ids[t + 1] if t + 1 < len(active_ids) else num_base_layers
    return Target(t, active_ids[t], nxt - 1)


@contextmanager
def truncated(model, n):
    layers, count = model.model.layers, model.config.num_hidden_layers
    model.model.layers, model.config.num_hidden_layers = layers[:n], n
    try:
        yield
    finally:
        model.model.layers, model.config.num_hidden_layers = layers, count


def encode_pair(row, tokenizer, lang, src_col, tgt_col, max_len):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(lang=lang, text=row[src_col])}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(row[tgt_col], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {
        "input_ids": (prompt_ids + resp_ids)[:max_len],
        "labels": ([-100] * len(prompt_ids) + resp_ids)[:max_len],
    }


def build_recovery_data(tokenizer, sources, src_col, tgt_col, cfg):
    train_parts, eval_parts = [], []
    for path, lang in sources:
        ds = read_tsv(path)
        ds = ds.shuffle(seed=cfg.seed).select(range(min(len(ds), cfg.train_per_lang + cfg.eval_per_lang)))
        ds = ds.map(
            encode_pair, remove_columns=ds.column_names,
            fn_kwargs=dict(tokenizer=tokenizer, lang=lang, src_col=src_col, tgt_col=tgt_col, max_len=cfg.max_len),
        )
        eval_parts.append(ds.select(range(cfg.eval_per_lang)))
        train_parts.append(ds.select(range(cfg.eval_per_lang, len(ds))))
    return concatenate_datasets(train_parts), concatenate_datasets(eval_parts)


def make_collator(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            mask[i, :k] = 1
            labels[i, :k] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return collate


class BestLayerCallback(TrainerCallback):
    def __init__(self, layer, patience):
        self.layer, self.patience = layer, patience
        self.best, self.best_state, self.bad, self.history = None, None, 0, []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.history.append(dict(metrics))
        if self.best is None or metrics["eval_loss"] < self.best["eval_loss"]:
            self.best, self.bad = dict(metrics), 0
            self.best_state = {k: v.detach().clone().cpu() for k, v in self.layer.state_dict().items()}
        else:
            self.bad += 1
            if self.bad >= self.patience:
                control.should_training_stop = True

    def restore(self):
        if self.best_state is not None:
            self.layer.load_state_dict(self.best_state)


class PruneRecoveryTrainer(Trainer):
    loss_is_scaled_for_ga = False

    def __init__(self, *args, teacher, target, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.teacher, self.target, self.cfg = teacher, target, cfg
        self.scales = dict.fromkeys(TERMS, 1.0)
        self._cap = {}
        self._acc = {"train": defaultdict(list), "eval": defaultdict(list)}
        student = self.accelerator.unwrap_model(self.model)
        self._amp_dtype = student.model.embed_tokens.weight.dtype
        self._handles = [
            self._capture(student.model.layers[target.trainable_idx], "student"),
            self._capture(teacher.model.layers[target.base_idx], "teacher"),
        ]

    def _capture(self, module, key):
        def hook(mod, args, output):
            self._cap[key] = output[0] if isinstance(output, tuple) else output
        return module.register_forward_hook(hook)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def _autocast(self, device):
        enabled = device.type == "cuda" and self._amp_dtype in (torch.float16, torch.bfloat16)
        return torch.autocast(device.type, dtype=self._amp_dtype, enabled=enabled)

    def _hidden_mse(self, mask):
        student = self._cap["student"].float()
        teacher = self._cap["teacher"].to(student.device).float()
        diff = (student - teacher).pow(2)
        if self.cfg.normalize_mse:
            per_token = diff.sum(-1) / teacher.pow(2).sum(-1).clamp_min(1e-6)
        else:
            per_token = diff.mean(-1)
        return per_token[mask].mean()

    def _kl(self, student_logits, teacher_logits, labels, mask):
        if self.cfg.kl_on_response_only:
            pos = labels[:, 1:] != -100
        else:
            pos = mask[:, 1:] & mask[:, :-1]
        if not pos.any():
            return student_logits.new_zeros((), dtype=torch.float32)
        temp = self.cfg.temperature
        s = student_logits[:, :-1][pos].float() / temp
        t = teacher_logits[:, :-1][pos].float() / temp
        kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean")
        return kl * temp ** 2

    def _ce(self, logits, labels):
        pos = labels[:, 1:] != -100
        if not pos.any():
            return logits.new_zeros((), dtype=torch.float32)
        return F.cross_entropy(logits[:, :-1][pos].float(), labels[:, 1:][pos])

    def _terms(self, model, inputs):
        student = self.accelerator.unwrap_model(model)
        labels = inputs["labels"]
        batch = {k: v for k, v in inputs.items() if k != "labels"}
        teacher_batch = {k: v.to(self.teacher.device) for k, v in batch.items()}
        mask = batch["attention_mask"].bool()
        t, b = self.target.trainable_idx, self.target.base_idx

        with self._autocast(labels.device):
            with torch.no_grad():
                if self.cfg.kl_weight > 0:
                    teacher_logits = self.teacher(**teacher_batch, use_cache=False).logits.to(labels.device)
                else:
                    with truncated(self.teacher, b + 1):
                        self.teacher.model(**teacher_batch, use_cache=False)
            if self.cfg.needs_logits:
                logits = student(**batch, use_cache=False).logits
            else:
                with truncated(student, t + 1):
                    student.model(**batch, use_cache=False)

        terms = {"mse": self._hidden_mse(mask)}
        if self.cfg.kl_weight > 0:
            terms["kl"] = self._kl(logits, teacher_logits, labels, mask)
        if self.cfg.ce_weight > 0:
            terms["ce"] = self._ce(logits, labels)
        return terms

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        terms = self._terms(model, inputs)
        loss = sum(self.cfg.weight(k) * v / self.scales[k] for k, v in terms.items())
        acc = self._acc["train" if model.training else "eval"]
        for k, v in terms.items():
            acc[k].append(v.item())
        return (loss, {k: v.detach() for k, v in terms.items()}) if return_outputs else loss

    @torch.no_grad()
    def estimate_scales(self):
        # each term is divided by its starting magnitude, so weights express relative importance
        gen = torch.Generator().manual_seed(self.cfg.seed)
        loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                            generator=gen, collate_fn=self.data_collator)
        device = self.model.device
        sums, n = defaultdict(float), 0
        for inputs in itertools.islice(loader, self.cfg.scale_batches):
            for k, v in self._terms(self.model, {k: x.to(device) for k, x in inputs.items()}).items():
                sums[k] += v.item()
            n += 1
        self.scales.update({k: max(s / n, 1e-8) for k, s in sums.items()})

    def log(self, logs, *args, **kwargs):
        split = "eval" if "eval_loss" in logs else "train" if "loss" in logs else None
        if split and self._acc[split]:
            for k, vals in self._acc[split].items():
                logs[f"{split}_{k}"] = sum(vals) / len(vals)
            self._acc[split] = defaultdict(list)
        super().log(logs, *args, **kwargs)


def heal_pruned_model_kd(model, teacher, tokenizer, iteration, p, target, train_ds, eval_ds, cfg):
    dtype = model.model.embed_tokens.weight.dtype
    layer = model.model.layers[target.trainable_idx]
    on_gpu = torch.cuda.is_available()
    model.requires_grad_(False)
    layer.float().requires_grad_(True)

    total_steps = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))

    args = TrainingArguments(
        output_dir=str(p / f"recovery_iter{iteration}"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        eval_on_start=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        prediction_loss_only=True,
        fp16=on_gpu and dtype == torch.float16,
        bf16=on_gpu and dtype == torch.bfloat16,
        seed=cfg.seed,
    )
    callback = BestLayerCallback(layer, cfg.patience)
    trainer = PruneRecoveryTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id), callbacks=[callback],
        teacher=teacher, target=target, cfg=cfg,
    )
    if cfg.balance_terms:
        trainer.estimate_scales()

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30 if on_gpu else float("nan")

    callback.restore()
    trainer.remove_hooks()
    # drop the autocast wrapper Accelerate leaves on model.forward
    trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    layer.to(dtype).requires_grad_(False)
    model.eval()

    first, best = callback.history[0], callback.best
    row = {
        "iteration": iteration,
        "trainable_layer": target.trainable_orig,
        "base_target_layer": target.base_idx,
        "eval_loss_before": first["eval_loss"],
        "eval_loss_best": best["eval_loss"],
    }
    for k in TERMS:
        row[f"w_{k}"] = cfg.weight(k)
        row[f"scale_{k}"] = trainer.scales[k] if f"eval_{k}" in first else None
        row[f"{k}_before"] = first.get(f"eval_{k}")
        row[f"{k}_best"] = best.get(f"eval_{k}")
    row.update(
        steps=trainer.state.global_step, seconds=seconds, peak_gib=peak_gib,
        trainable_params=sum(x.numel() for x in layer.parameters()),
    )
    log_path = p / "recovery_log.tsv"
    pd.DataFrame([row]).to_csv(log_path, sep="\t", mode="a", header=not log_path.exists(), index=False)
    return model
