import argparse
from pathlib import Path
import pandas as pd
from metrics import get_metrics


def process_file(path, metrics):
    df = pd.read_csv(path, sep="\t")
    srcs = df["src"].astype(str).tolist()
    refs = df["tgt"].astype(str).tolist()
    mts = df["mt"].astype(str).tolist()

    corpus_scores = {}
    for metric in metrics:
        print(f"Running for {metric}")
        sent_scores, corpus_score = metric.compute(srcs, mts, refs)
        df[f"{metric.name}_score"] = sent_scores
        corpus_scores[metric.name] = corpus_score

    df.to_csv(path, sep="\t", index=False)
    return corpus_scores


def main(file_path, metric_names):
    tsv_file = Path(file_path)
    metrics = get_metrics(metric_names)

    model_name = tsv_file.parent.name
    lang_pair = tsv_file.stem

    print(f"|> Evaluating {model_name}/{tsv_file.name}")
    scores = process_file(tsv_file, metrics)

    summary_rows = [{"model": model_name, "lang_pair": lang_pair, **scores}]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = tsv_file.parent / f"{tsv_file.stem}_evaluation_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"|> Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name", required=True, help="path to the specific TSV file"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="subset of: bleu chrf++ comet",
    )
    args = parser.parse_args()
    main(args.file_name, args.metrics)