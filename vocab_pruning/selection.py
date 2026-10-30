from collections import Counter

from config import LANGS, SCRIPT_RANGES


def count_tokens(codec, src_lines, tgt_lines, src, tgt):
    src_counts, tgt_counts = Counter(), Counter()
    for line in src_lines:
        src_counts.update(codec.encode_src(line, src, tgt))
    for line in tgt_lines:
        tgt_counts.update(codec.encode_tgt(line, tgt))
    return src_counts, tgt_counts


def select_ids(counts, min_count=1, top_k=None, coverage=None):
    ranked = [(i, c) for i, c in counts.most_common() if c >= min_count]
    if coverage is not None:
        total, running, cut = sum(c for _, c in ranked), 0, []
        for i, c in ranked:
            cut.append((i, c))
            running += c
            if running / total >= coverage:
                break
        ranked = cut
    if top_k is not None:
        ranked = ranked[:top_k]
    return {i for i, _ in ranked}


def script_char_ids(vocab, scripts):
    ranges = [r for s in scripts for r in SCRIPT_RANGES[s]]
    ids = set()
    for tok, i in vocab.items():
        core = tok[1:] if tok.startswith("▁") else tok
        if tok == "▁" or (len(core) == 1 and any(a <= ord(core) <= b for a, b in ranges)):
            ids.add(i)
    return ids


def build_keep_sets(codec, setting, src, tgts, counts, gen_counts, mode):
    sel = {k: setting[k] for k in ["min_count", "top_k", "coverage"]}

    keep_src = select_ids(sum((counts[t][0] for t in tgts), Counter()), **sel)
    for t in tgts:
        keep_src |= codec.required_src_ids(src, t)
    if setting["char_fallback"]:
        keep_src |= script_char_ids(codec.src_vocab, ["common"] + LANGS[src]["scripts"])

    keep_tgt, per_lang = set(), {}
    for t in tgts:
        ids = select_ids(counts[t][1], **sel) | codec.required_tgt_ids(t)
        if setting["char_fallback"]:
            ids |= script_char_ids(codec.tgt_vocab, ["common", "latin"] + LANGS[t]["scripts"])
        if setting["model_outputs"]:
            ids |= set(gen_counts[t])
        per_lang[t] = len(ids)
        keep_tgt |= ids

    if mode == "union":
        keep_src = keep_tgt = keep_src | keep_tgt
    return sorted(keep_src), sorted(keep_tgt), per_lang
