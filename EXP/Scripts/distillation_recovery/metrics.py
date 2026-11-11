from abc import ABC, abstractmethod

from sacrebleu.metrics import BLEU, CHRF


class Metric(ABC):
    name = "metric"

    @abstractmethod
    def compute(self, srcs, mts, refs):
        # returns (list_of_sentence_scores, corpus_score)
        raise NotImplementedError


class BLEUMetric(Metric):
    name = "bleu"

    # flores200 (SPM) tokenizer: the same one the inference script used for _summary.tsv.
    # The default "13a" tokenizer is built for English and under-segments Indic scripts.
    # The scorer objects are built ONCE: sacrebleu.sentence_bleu() builds a new BLEU object
    # (and reloads the SentencePiece model) on every call, i.e. once per sentence.
    def __init__(self, tokenize="flores200"):
        self.corpus = BLEU(tokenize=tokenize)
        self.sentence = BLEU(tokenize=tokenize, effective_order=True)

    def compute(self, srcs, mts, refs):
        sent_scores = [self.sentence.sentence_score(mt, [ref]).score for mt, ref in zip(mts, refs)]
        corpus_score = self.corpus.corpus_score(mts, [refs]).score
        return sent_scores, corpus_score


class ChrFMetric(Metric):
    name = "chrf++"

    def __init__(self):
        self.scorer = CHRF(word_order=2)

    def compute(self, srcs, mts, refs):
        sent_scores = [self.scorer.sentence_score(mt, [ref]).score for mt, ref in zip(mts, refs)]
        corpus_score = self.scorer.corpus_score(mts, [refs]).score
        return sent_scores, corpus_score


class _NeuralMetric(Metric):
    """Shared predict() for COMET-architecture models (COMET, COMTAIL).

    num_workers=0 loads batches in the main process. COMET's default can start
    DataLoader worker processes, and each one adds its own RAM on top of the main
    process, which is what a Slurm --mem limit counts.
    """
    def _setup(self, model, batch_size, num_workers):
        import torch
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpus = 1 if torch.cuda.is_available() else 0

    def _predict(self, data):
        output = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus,
                                    num_workers=self.num_workers, progress_bar=True)
        return output.scores, output.system_score


class COMETMetric(_NeuralMetric):
    name = "comet"

    def __init__(self, model_name="Unbabel/wmt22-comet-da", batch_size=16, num_workers=0):
        from comet import download_model, load_from_checkpoint
        self._setup(load_from_checkpoint(download_model(model_name)), batch_size, num_workers)

    def compute(self, srcs, mts, refs):
        return self._predict([{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)])


class _COMTAILBase(_NeuralMetric):
    """COMTAIL models are COMET-architecture checkpoints, loaded from a local .ckpt file.
    The checkpoint is loaded once in __init__ and reused for every file."""
    ckpt_subpath = None   # set in subclasses

    def __init__(self, comtail_dir="comtail-models", ckpt_path=None, batch_size=16, num_workers=0):
        from pathlib import Path
        from comet import load_from_checkpoint
        path = Path(ckpt_path) if ckpt_path else Path(comtail_dir) / self.ckpt_subpath
        if not path.exists():
            raise FileNotFoundError(f"{self.name}: checkpoint not found at {path}")
        self._setup(load_from_checkpoint(str(path)), batch_size, num_workers)


class COMTAILDAMetric(_COMTAILBase):
    """Reference-based: uses src, mt and ref."""
    name = "comtail-da"
    ckpt_subpath = "COMTAIL-DA/checkpoints/epoch=4-step=230265.ckpt"

    def compute(self, srcs, mts, refs):
        return self._predict([{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)])


class COMTAILQEMetric(_COMTAILBase):
    """Reference-free quality estimation: uses only src and mt (refs are ignored)."""
    name = "comtail-qe"
    ckpt_subpath = "COMTAIL-QE/checkpoints/epoch=4-step=230265.ckpt"

    def compute(self, srcs, mts, refs=None):
        return self._predict([{"src": s, "mt": m} for s, m in zip(srcs, mts)])


REGISTRY = {
    "bleu": BLEUMetric,
    "chrf++": ChrFMetric,
    "comet": COMETMetric,
    "comtail-da": COMTAILDAMetric,
    "comtail-qe": COMTAILQEMetric,
}
NEURAL_METRICS = {"comet", "comtail-da", "comtail-qe"}


def build_metric(name, comet_kwargs=None, comtail_kwargs=None):
    if name not in REGISTRY:
        raise ValueError(f"unknown metric '{name}', choose from {list(REGISTRY)}")
    if name == "comet":
        return REGISTRY[name](**(comet_kwargs or {}))
    if name.startswith("comtail"):
        return REGISTRY[name](**(comtail_kwargs or {}).get(name, {}))
    return REGISTRY[name]()


def get_metrics(names=None, comet_kwargs=None, comtail_kwargs=None):
    """Build all metrics at once (kept for compatibility; evaluate.py builds them one at a time)."""
    names = names or list(REGISTRY.keys())
    return [build_metric(n, comet_kwargs, comtail_kwargs) for n in names]
