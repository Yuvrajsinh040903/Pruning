import os
os.environ["HF_HOME"] = "/scratch/yuvraj.bodana"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yuvraj.bodana"
import torch

import config as C
from bhashaverse import Codec, load_model, save_model
from data import encode_pairs, load_train_dev
from distill import train
from evaluate import evaluate_model, write_results
from student import build_student, kept_layers, param_count


def main():
    codec = Codec(C.TEACHER_PATH, C.FLORES_CODES, C.MAX_SRC_LEN, C.MAX_TGT_LEN)

    teacher = load_model(C.TEACHER_PATH, torch.float32)
    enc_keep, dec_keep = kept_layers(teacher.config, C)
    print(f"encoder: keep {len(enc_keep)}/{teacher.config.encoder_layers} {enc_keep}")
    print(f"decoder: keep {len(dec_keep)}/{teacher.config.decoder_layers} {dec_keep}")
    run_dir = os.path.join(C.OUTPUT_DIR, C.RUN_NAME or f"{C.LAYER_RULE}_enc{len(enc_keep)}_dec{len(dec_keep)}")
    os.makedirs(run_dir, exist_ok=True)
    meta = {"teacher": C.TEACHER_PATH, "layer_rule": C.LAYER_RULE, "encoder_layers_kept": enc_keep,
            "decoder_layers_kept": dec_keep}

    init_dir, best_dir = os.path.join(run_dir, "init"), os.path.join(run_dir, "best")
    student = build_student(teacher, enc_keep, dec_keep, C.STUDENT_DROPOUT)
    print(f"parameters: teacher {param_count(teacher):,} -> student {param_count(student):,}")
    if not os.path.exists(init_dir):
        save_model(student, init_dir, C.TEACHER_PATH, meta)
    teacher = teacher.to(C.TEACHER_DTYPE)

    rows = []
    if C.EVALUATE_TEACHER:
        rows += evaluate_model(teacher, codec, "teacher", os.path.join(C.OUTPUT_DIR, "teacher"))
        teacher.to("cpu")
    if C.EVALUATE_BEFORE_DISTILL:
        rows += evaluate_model(load_model(init_dir, C.EVAL_DTYPE), codec, "pruned_no_distill",
                               os.path.join(run_dir, "eval_init"))

    if os.path.exists(os.path.join(best_dir, "model.safetensors")):
        print(f"{best_dir} exists, skipping distillation")
    else:
        train_examples, dev_examples = [], []
        for tgt in C.TGT_LANGS:
            (src, ref), (dev_src, dev_ref) = load_train_dev(tgt)
            train_examples += encode_pairs(codec, src, ref, tgt)
            dev_examples += encode_pairs(codec, dev_src, dev_ref, tgt)
            print(f"  eng-{tgt}: {len(src):,} train / {len(dev_src):,} dev pairs")
        train(student, teacher, codec, train_examples, dev_examples, enc_keep, dec_keep, run_dir, meta)
    del student, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows += evaluate_model(load_model(best_dir, C.EVAL_DTYPE), codec, "distilled", os.path.join(run_dir, "eval_best"))
    write_results(os.path.join(run_dir, "results.tsv"), rows)


if __name__ == "__main__":
    main()
