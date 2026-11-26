import time
import torch
import pandas as pd
from dataclasses import dataclass
from peft import LoraConfig, OFTConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments
from recovery_kd import make_collator

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class PeftRecoveryConfig:
    method: str = "lora"  # "lora" or "oft"
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 4
    eval_steps: int = 100
    patience: int = 2
    optim: str = "paged_adamw_32bit"
    max_len: int = 512
    train_per_lang: int = 4000
    eval_per_lang: int = 50
    seed: int = 0


def peft_config(method):
    if method == "lora":
        return LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, bias="none", use_rslora=True,
                          task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    if method == "oft":
        return OFTConfig(r=32, oft_block_size=0, task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    raise ValueError(f"unknown method {method}")


class BestAdapterCallback(TrainerCallback):
    def __init__(self, model, patience):
        self.model, self.patience = model, patience
        self.best, self.best_state, self.bad, self.history = None, None, 0, []

    def _trainable(self):
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = metrics["eval_loss"]
        self.history.append(loss)
        if self.best is None or loss < self.best:
            self.best, self.bad = loss, 0
            self.best_state = {n: p.detach().clone().cpu() for n, p in self._trainable().items()}
        else:
            self.bad += 1
            if self.bad >= self.patience:
                control.should_training_stop = True

    def restore(self):
        if self.best_state is None:
            return
        with torch.no_grad():
            for n, p in self._trainable().items():
                p.copy_(self.best_state[n].to(p.device, p.dtype))


def heal_pruned_model_peft(model, tokenizer, iteration, p, train_ds, eval_ds, cfg):
    dtype = model.model.embed_tokens.weight.dtype
    on_gpu = torch.cuda.is_available()
    model.requires_grad_(False)
    model = get_peft_model(model, peft_config(cfg.method))
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)

    args = TrainingArguments(
        output_dir=str(p / f"recovery_iter{iteration}"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type="cosine",
        optim=cfg.optim,
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
    callback = BestAdapterCallback(model, cfg.patience)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id), callbacks=[callback],
    )

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30 if on_gpu else float("nan")

    callback.restore()
    # drop the autocast wrapper Accelerate leaves on model.forward
    trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    model = model.merge_and_unload().to(dtype)
    model.requires_grad_(False)
    model.eval()

    row = {
        "iteration": iteration,
        "method": cfg.method,
        "ce_before": callback.history[0],
        "ce_best": callback.best,
        "steps": trainer.state.global_step,
        "seconds": seconds,
        "peak_gib": peak_gib,
        "trainable_params": trainable,
    }
    log_path = p / "recovery_log.tsv"
    pd.DataFrame([row]).to_csv(log_path, sep="\t", mode="a", header=not log_path.exists(), index=False)
    del trainer
    return model
