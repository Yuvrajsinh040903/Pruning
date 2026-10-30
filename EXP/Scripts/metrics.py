from abc import ABC, abstractmethod
import sacrebleu


class Metric(ABC):
    name = "metric"

    @abstractmethod
    def compute(self, srcs, mts, refs):
        # returns (list_of_sentence_scores, corpus_score)
        raise NotImplementedError


class BLEUMetric(Metric):
    name = "bleu"

    def compute(self, srcs, mts, refs):
        sent_scores = [sacrebleu.sentence_bleu(mt, [ref]).score for mt, ref in zip(mts, refs)]
        corpus_score = sacrebleu.corpus_bleu(mts, [refs]).score
        return sent_scores, corpus_score


class ChrFMetric(Metric):
    name = "chrf++"

    def compute(self, srcs, mts, refs):
        sent_scores = [sacrebleu.sentence_chrf(mt, [ref], word_order=2).score for mt, ref in zip(mts, refs)]
        corpus_score = sacrebleu.corpus_chrf(mts, [refs], word_order=2).score
        return sent_scores, corpus_score


class COMETMetric(Metric):
    name = "comet"

    def __init__(self, model_name="Unbabel/wmt22-comet-da", batch_size=16):
        import torch
        from comet import download_model, load_from_checkpoint
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        self.batch_size = batch_size
        self.gpus = 1 if torch.cuda.is_available() else 0

    def compute(self, srcs, mts, refs):
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
        output = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
        return output.scores, output.system_score


REGISTRY = {
    "bleu": lambda **kw: BLEUMetric(),
    "chrf++": lambda **kw: ChrFMetric(),
    "comet": lambda **kw: COMETMetric(**kw),
}


def get_metrics(names=None, comet_kwargs=None):
    comet_kwargs = comet_kwargs or {}
    names = names or list(REGISTRY.keys())
    metrics = []
    for n in names:
        kwargs = comet_kwargs if n == "comet" else {}
        metrics.append(REGISTRY[n](**kwargs))
    return metrics