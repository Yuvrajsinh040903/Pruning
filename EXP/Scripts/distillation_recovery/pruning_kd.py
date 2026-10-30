import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass
from datasets import load_dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from pruning_core import PROMPT, reindex_layers


@dataclass
class RecoveryConfig:
    use_kl: bool = False
    kl_weight: float = 1.0
    temperature: float = 1.0
    mse_weight: float = 1.0
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
    eval_size: int = 200
    seed: int = 0


@dataclass
class Gap:
    removed: nn.Module
    frozen: nn.Module
    trainable_idx: int
    removed_first: bool

    @property
    def target(self):
        return self.frozen if self.removed_first else self.removed

    def pair(self):
        return [self.removed, self.frozen] if self.removed_first else [self.frozen, self.removed]


def make_gap(model, removed, removed_idx):
    removed_first = removed_idx == 0
    idx = 0 if removed_first else removed_idx - 1
    frozen = copy.deepcopy(model.model.layers[idx])
    for module in (removed, frozen):
        module.requires_grad_(False)
        module.eval()
    return Gap(removed, frozen, idx, removed_first)


@contextmanager
def swapped_layers(model, layers):
    original = list(model.model.layers)
    model.model.layers = nn.ModuleList(layers)
    reindex_layers(model)
    try:
        yield
    finally:
        model.model.layers = nn.ModuleList(original)
        reindex_layers(model)


def build_recovery_data(tokenizer, path, src_col, tgt_col, cfg):
    ds = load_dataset("csv", data_files=path, delimiter="\t", split="train")

    def encode(row):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=row[src_col])}],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(row[tgt_col], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        return {
            "input_ids": (prompt_ids + resp_ids)[:cfg.max_len],
            "labels": ([-100] * len(prompt_ids) + resp_ids)[:cfg.max_len],
        }

    ds = ds.map(encode, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=cfg.eval_size, seed=cfg.seed)
    return split["train"], split["test"]


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
        self.best, self.best_state, self.bad, self.history = float("inf"), None, 0, []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = metrics["eval_loss"]
        self.history.append(loss)
        if loss < self.best:
            self.best, self.bad = loss, 0
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

    def __init__(self, *args, gap, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.gap, self.cfg = gap, cfg
        self._cap = {}
        self._acc = {"mse": [], "kl": []}
        base = self.accelerator.unwrap_model(self.model)
        self._handles = [
            self._capture(base.model.layers[gap.trainable_idx], "student"),
            self._capture(gap.target, "teacher"),
        ]

    def _capture(self, module, key):
        def hook(mod, args, output):
            self._cap[key] = output[0] if isinstance(output, tuple) else output
        return module.register_forward_hook(hook)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def _teacher_layers(self, student_layers):
        t = self.gap.trainable_idx
        return student_layers[:t] + self.gap.pair() + student_layers[t + 1:]

    def _hidden_mse(self, mask):
        student = self._cap["student"].float()
        teacher = self._cap["teacher"].float()
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
            return student_logits.new_zeros(())
        temp = self.cfg.temperature
        s = student_logits[:, :-1][pos].float() / temp
        t = teacher_logits[:, :-1][pos].float() / temp
        kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean")
        return kl * temp ** 2

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = self.accelerator.unwrap_model(model)
        labels = inputs["labels"]
        batch = {k: v for k, v in inputs.items() if k != "labels"}
        mask = batch["attention_mask"].bool()
        layers = list(base.model.layers)
        t = self.gap.trainable_idx
        teacher_layers = self._teacher_layers(layers)
        kl = torch.zeros((), device=labels.device)

        if self.cfg.use_kl:
            with torch.no_grad(), swapped_layers(base, teacher_layers):
                teacher_out = base(**batch, use_cache=False)
            student_out = base(**batch, use_cache=False)
            mse = self._hidden_mse(mask)
            kl = self._kl(student_out.logits, teacher_out.logits, labels, mask)
            loss = self.cfg.mse_weight * mse + self.cfg.kl_weight * kl
        else:
            with torch.no_grad(), swapped_layers(base, teacher_layers[:t + 2]):
                base.model(**batch, use_cache=False)
            with swapped_layers(base, layers[:t + 1]):
                base.model(**batch, use_cache=False)
            mse = self._hidden_mse(mask)
            loss = self.cfg.mse_weight * mse

        if model.training:
            self._acc["mse"].append(mse.detach().item())
            self._acc["kl"].append(kl.detach().item())
        return (loss, {"mse": mse.detach(), "kl": kl.detach()}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._acc["mse"]:
            logs["train_mse"] = sum(self._acc["mse"]) / len(self._acc["mse"])
            if self.cfg.use_kl:
                logs["train_kl"] = sum(self._acc["kl"]) / len(self._acc["kl"])
            self._acc = {"mse": [], "kl": []}
        super().log(logs, *args, **kwargs)


def heal_pruned_model_kd(model, tokenizer, iteration, p, gap, train_ds, eval_ds, cfg):
    layer = model.model.layers[gap.trainable_idx]
    dtype = next(model.parameters()).dtype
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
        gap=gap, cfg=cfg,
    )

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30 if on_gpu else float("nan")

    callback.restore()
    trainer.remove_hooks()
    layer.to(dtype).requires_grad_(False)
    model.eval()

    row = {
        "iteration": iteration,
        "use_kl": cfg.use_kl,
        "eval_loss_before": callback.history[0],
        "eval_loss_best": callback.best,
        "steps": trainer.state.global_step,
        "seconds": seconds,
        "peak_gib": peak_gib,
        "trainable_params": sum(x.numel() for x in layer.parameters()),
    }
    log_path = p / "recovery_log.tsv"
    pd.DataFrame([row]).to_csv(log_path, sep="\t", mode="a", header=not log_path.exists(), index=False)
    return model