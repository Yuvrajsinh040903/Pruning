import json
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import config as C
from bhashaverse import save_model
from data import collate, token_batches


def hidden_pairs(s_states, t_states, keep):
    if C.COS_LAYERS == "last":
        return [(s_states[-1], t_states[-1])]
    # hidden_states[0] is the embedding output, hidden_states[i + 1] the output of layer i
    return [(s_states[j + 1], t_states[i + 1]) for j, i in enumerate(keep)]


def cosine_loss(pairs, mask):
    losses = [(1 - F.cosine_similarity(s[mask].float(), t[mask].float(), dim=-1)).mean() for s, t in pairs]
    return sum(losses) / len(losses)


def distill_losses(student, teacher, batch, enc_keep, dec_keep):
    labels = batch.pop("labels")
    need_hidden = C.ALPHA_COS > 0
    with torch.no_grad():
        t_out = teacher(**batch, output_hidden_states=need_hidden)
    s_out = student(**batch, output_hidden_states=need_hidden)

    mask = labels != -100
    T = C.TEMPERATURE
    s_logits, t_logits = s_out.logits[mask].float(), t_out.logits[mask].float()
    parts = {"kd": F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1),
                            log_target=True, reduction="batchmean") * T * T,
             "ce": F.cross_entropy(s_logits, labels[mask])}
    if need_hidden:
        cos = []
        if "encoder" in C.COS_STACKS:
            pairs = hidden_pairs(s_out.encoder_hidden_states, t_out.encoder_hidden_states, enc_keep)
            cos.append(cosine_loss(pairs, batch["attention_mask"].bool()))
        if "decoder" in C.COS_STACKS:
            cos.append(cosine_loss(hidden_pairs(s_out.decoder_hidden_states, t_out.decoder_hidden_states, dec_keep), mask))
        parts["cos"] = sum(cos) / len(cos)
    else:
        parts["cos"] = torch.zeros((), device=s_logits.device)
    parts["total"] = C.ALPHA_KD * parts["kd"] + C.ALPHA_CE * parts["ce"] + C.ALPHA_COS * parts["cos"]
    return parts


def amp_context():
    return torch.autocast("cuda", dtype=C.AMP_DTYPE) if C.DEVICE.startswith("cuda") else nullcontext()


@torch.no_grad()
def dev_losses(student, teacher, codec, dev_examples, enc_keep, dec_keep):
    student.eval()
    sums, tokens = {}, 0
    for idx in token_batches(dev_examples, C.MAX_TOKENS_PER_BATCH, shuffle=False):
        batch = collate(dev_examples, idx, codec, C.DEVICE)
        n = int((batch["labels"] != -100).sum())
        with amp_context():
            parts = distill_losses(student, teacher, batch, enc_keep, dec_keep)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * n
        tokens += n
    student.train()
    return {k: v / tokens for k, v in sums.items()}


def freeze_embeddings(model):
    m = model.model
    for module in [m.shared, m.encoder.embed_tokens, m.decoder.embed_tokens, m.encoder.embed_positions,
                   m.decoder.embed_positions, model.lm_head]:
        for p in module.parameters():
            p.requires_grad = False


def make_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": C.WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=C.LR, betas=(0.9, 0.98), eps=1e-6)

    def schedule(step):
        if step < C.WARMUP_STEPS:
            return (step + 1) / C.WARMUP_STEPS
        return max(0.0, (C.MAX_STEPS - step) / max(1, C.MAX_STEPS - C.WARMUP_STEPS))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def train(student, teacher, codec, train_examples, dev_examples, enc_keep, dec_keep, run_dir, meta):
    torch.manual_seed(C.SEED)
    student.to(C.DEVICE).train()
    teacher.to(C.DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if C.FREEZE_EMBEDDINGS:
        freeze_embeddings(student)
    if C.GRADIENT_CHECKPOINTING:
        student.config.use_cache = False
        student.gradient_checkpointing_enable()
    opt, sched = make_optimizer(student)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  trainable parameters: {trainable:,}")

    log_path = os.path.join(run_dir, "train_log.jsonl")
    best_dir = os.path.join(run_dir, "best")
    best = math.inf
    step, micro, epoch = 0, 0, 0
    running, start = {}, time.time()

    def evaluate_and_save():
        nonlocal best
        dev = dev_losses(student, teacher, codec, dev_examples, enc_keep, dec_keep)
        improved = dev[C.SELECT_BEST_BY] < best
        if improved:
            best = dev[C.SELECT_BEST_BY]
            save_model(student, best_dir, C.TEACHER_PATH, {**meta, "step": step, "dev": dev})
        record = {"step": step, "dev": dev, "best": improved}
        print(f"  [dev] step {step} " + " ".join(f"{k}={v:.4f}" for k, v in dev.items()) + (" *" if improved else ""))
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    evaluate_and_save()
    while step < C.MAX_STEPS:
        for idx in token_batches(train_examples, C.MAX_TOKENS_PER_BATCH, shuffle=True, seed=C.SEED + epoch):
            batch = collate(train_examples, idx, codec, C.DEVICE)
            with amp_context():
                parts = distill_losses(student, teacher, batch, enc_keep, dec_keep)
            (parts["total"] / C.GRAD_ACCUM).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item() / C.GRAD_ACCUM
            micro += 1
            if micro % C.GRAD_ACCUM:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], C.MAX_GRAD_NORM)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % C.LOG_EVERY == 0:
                avg = {k: v / C.LOG_EVERY for k, v in running.items()}
                rate = C.LOG_EVERY / (time.time() - start)
                print(f"  step {step}/{C.MAX_STEPS} lr={sched.get_last_lr()[0]:.2e} gnorm={float(grad_norm):.2f} "
                      + " ".join(f"{k}={v:.4f}" for k, v in avg.items()) + f" ({rate:.2f} steps/s)")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "train": avg, "lr": sched.get_last_lr()[0]}) + "\n")
                running, start = {}, time.time()
            if step % C.EVAL_EVERY == 0 or step == C.MAX_STEPS:
                evaluate_and_save()
            if step >= C.MAX_STEPS:
                break
        epoch += 1
    return best_dir
