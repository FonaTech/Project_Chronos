"""Document-mix translation shards: science-weighted subsets, merged n-in-1
samples up to a context limit, and bidirectional SFT records.

Why: ``shards_v1`` holds single sentences (SFT p50 75 / p99 225 tokens), so a
model trained on it never sees long inputs.  Here consecutive rows of the
same corpus unit are merged into one sample (2-in-1 … n-in-1).

Measured on ``corpus_v1/pairs`` (adjacent-pair vs random-pair content-word
overlap): most WMT19 units and iic keep document order (ratio 2–25), while
OPUS-100 and WMT19 unit 00012 are shuffled (ratio ≈ 1).  Each block is
classified here with the same statistic; merges inside an ordered block are
coherent discourse.  Merges of shuffled rows are random concatenations: they
teach length and position handling but not coherence, and the manifest
counts them separately.  Document boundaries are not stored in the corpus,
so some ordered merges cross a document boundary.

Layout of ``out_dir`` (same readers as ``shards.py`` plus one extra file):
  pretrain/{train,valid}/*.bin       flat tokens, ``<|endoftext|>``-separated
  sft/{train,valid}/<stem>.bin/.idx/.plen   as in shards.py
  sft/{train,valid}/<stem>.dir       uint8 per record: 0 = en→zh, 1 = zh→en
  sft/{train,valid}/<stem>.n         uint8 per record: sentences merged (1..255)
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from chronos_translate.corpus.shards import SFT_PROMPTS, _dtype_for

DIRECTIONS = ("en_zh", "zh_en")
DIR_CODE = {d: i for i, d in enumerate(DIRECTIONS)}
# Share of SAMPLES per merge size; "long" draws n uniformly in [6, n_max]
# where n_max is what fits the context limit.
DEFAULT_SHARES = {"1": 0.40, "2": 0.20, "3": 0.15, "4": 0.10, "5": 0.05, "long": 0.10}

# Science score = keyword hits per row (English: a trailing "*" is a prefix,
# otherwise a whole word with an optional plural "s"; Chinese: substrings).
# Generic UN/policy vocabulary (measure, general, physical, data protection)
# is avoided on purpose.  Chunks are ranked by mean score per row.
SCIENCE_EN = (
    "research*", "scien*", "experiment*", "laborator*", "clinic*", "patient", "disease", "medic*",
    "vaccin*", "virus", "viruses", "bacteri*", "gene", "genes", "genetic*", "genom*", "protein", "cell",
    "molecul*", "chemi*", "physics", "physicist", "quantum", "particle", "atom", "atomic", "nuclear",
    "energy", "solar", "battery", "batteries", "electron*", "electric*", "climate", "carbon", "emission",
    "ocean*", "geolog*", "ecolog*", "ecosystem", "species", "evolution*", "biolog*", "neuro*", "brain",
    "algorithm*", "comput*", "software", "robot*", "artificial intelligence", "machine learning",
    "engineer*", "technolog*", "satellite", "planet*", "astronom*", "telescope", "mathemat*", "equation",
    "statistic*", "hypothes*", "sensor", "semiconductor", "polymer", "catalys*", "reactor", "fusion",
    "laser", "optic*", "spectr*", "drug", "therap*", "diagnos*", "infection", "cancer", "tumou?r", "dna",
    "rna", "enzyme", "microb*", "fossil", "isotope", "radiation", "theorem", "physiolog*", "pharmac*",
)


def _en_pattern(term: str) -> str:
    if term.endswith("*"):
        return re.escape(term[:-1]) + r"\w*"
    body = term if "?" in term else re.escape(term)
    return body + r"s?\b"



SCIENCE_ZH = (
    "研究", "科学", "实验", "实验室", "临床", "患者", "疾病", "医学", "医疗", "健康", "疫苗", "病毒", "细菌",
    "基因", "蛋白", "细胞", "分子", "化学", "物理", "量子", "粒子", "原子", "核能", "能源", "太阳能", "电池",
    "电子", "气候", "碳", "排放", "海洋", "地质", "生态", "物种", "进化", "生物", "神经", "大脑", "算法",
    "计算", "软件", "数据", "网络", "机器人", "人工智能", "机器学习", "工程", "技术", "卫星", "太空", "行星",
    "天文", "望远镜", "数学", "方程", "统计", "理论", "假设", "测量", "传感", "材料", "半导体", "聚合物",
    "催化", "反应堆", "聚变", "激光", "光学", "光谱", "药物", "治疗", "诊断",
)
_SCI_EN_RE = re.compile(r"\b(?:" + "|".join(_en_pattern(s) for s in SCIENCE_EN) + r")")
_SCI_ZH_RE = re.compile("|".join(re.escape(s) for s in sorted(SCIENCE_ZH, key=len, reverse=True)))
# Adjacency statistic (same as the measurement that motivated this module).
_WORD = re.compile(r"[a-z]{4,}")
_STOP = frozenset((
    "that this with from have will been were they their there which would could about into than them these "
    "those what when where while also more most such only over other some very after before because being "
    "between both each every however should since through under until upon your said says year years").split())


def science_hits(en: str, zh: str) -> int:
    """Keyword hits of one pair (English side lower-cased, Chinese side as is)."""
    return len(_SCI_EN_RE.findall(en.lower())) + len(_SCI_ZH_RE.findall(zh))


def _bag(text: str) -> frozenset:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOP)


@dataclass
class DocMixSpec:
    fraction: float                       # share of train rows kept (0.15 = 15 %)
    ctx_len: int                          # SFT record cap = pretrain window (tokens)
    shares: dict = field(default_factory=lambda: dict(DEFAULT_SHARES))
    long_min: int = 6                     # "long" samples: n ~ U[long_min, n_cap]
    zh_en_share: float = 1 / 3            # en→zh : zh→en = 2 : 1 (SFT and tagged pretrain docs)
    mono_share: float = 0.5               # pretrain: monolingual docs; the rest are tagged pairs
    para_break: float = 0.15              # chance that a sentence boundary becomes a newline
    science_weights: tuple = (2.0, 1.2, 1.0)    # high / medium / other
    science_quantiles: tuple = (0.10, 0.20)     # chunk shares of the high / medium tier
    chunk_rows: int = 256                 # selection unit (consecutive rows of one part)
    block_rows: int = 65536               # adjacency classification window
    ordered_ratio: float = 2.0            # adjacent/random overlap ratio for "ordered"
    ordered_min_adj: float = 0.05         # ... and at least this adjacent overlap
    seed: int = 20261107


def _scan_part(args) -> dict:
    path, block_rows, seed = args
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["zh", "en"])
    zh, en = table["zh"].to_pylist(), table["en"].to_pylist()
    sci = np.fromiter((science_hits(e, z) for e, z in zip(en, zh)), dtype=np.int16, count=len(en))
    rng = random.Random(seed)
    blocks = []
    for b0 in range(0, len(en), block_rows):
        bags = [_bag(s) for s in en[b0:b0 + block_rows]]
        n = len(bags)
        adj = sum(1 for a, b in zip(bags, bags[1:]) if a & b)
        k = min(20000, 2 * n) if n > 1 else 0
        rnd = sum(1 for _ in range(k) if bags[rng.randrange(n)] & bags[rng.randrange(n)])
        blocks.append((adj, max(0, n - 1), rnd, k))
    return {"name": os.path.basename(path), "rows": len(en), "sci": sci,
            "blocks": np.asarray(blocks, dtype=np.int64).reshape(-1, 4)}


def scan_corpus(pairs_dir: str, cache_dir: str, *, block_rows: int = 65536, workers: int = 8,
                log=print) -> dict:
    """Science hits per row and adjacency counts per block, cached in ``cache_dir``."""
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "scan.json")
    names = sorted(p for p in os.listdir(pairs_dir) if p.endswith(".parquet"))
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("names") == names and meta.get("block_rows") == block_rows:
            data = np.load(os.path.join(cache_dir, "scan.npz"))
            return {"names": names, "block_rows": block_rows, "rows": meta["rows"],
                    "sci": {n: data[f"sci/{n}"] for n in names}, "blocks": {n: data[f"blk/{n}"] for n in names}}
    t0 = time.time()
    jobs = [(os.path.join(pairs_dir, n), block_rows, 17 + i) for i, n in enumerate(names)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = {r["name"]: r for r in pool.map(_scan_part, jobs)}
    out = {"names": names, "block_rows": block_rows, "rows": {n: results[n]["rows"] for n in names},
           "sci": {n: results[n]["sci"] for n in names}, "blocks": {n: results[n]["blocks"] for n in names}}
    np.savez(os.path.join(cache_dir, "scan.npz"), **{f"sci/{n}": out["sci"][n] for n in names},
             **{f"blk/{n}": out["blocks"][n] for n in names})
    json.dump({"names": names, "block_rows": block_rows, "rows": out["rows"], "keywords_en": SCIENCE_EN,
               "keywords_zh": SCIENCE_ZH, "elapsed_s": round(time.time() - t0, 1)},
              open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[docmix] scanned {len(names)} parts in {time.time() - t0:.0f}s")
    return out


def block_ordered(blocks: np.ndarray, ratio: float, min_adj: float) -> np.ndarray:
    """Per block: adjacent-pair overlap ≥ ``ratio`` × random-pair overlap and ≥ ``min_adj``."""
    adj = blocks[:, 0] / np.maximum(blocks[:, 1], 1)
    rnd = blocks[:, 2] / np.maximum(blocks[:, 3], 1)
    return (adj >= ratio * np.maximum(rnd, 1e-3)) & (adj >= min_adj)


def select_chunks(scan: dict, spec: DocMixSpec) -> dict:
    """Science-weighted chunk sample of the train parts, without replacement.

    Chunks (``chunk_rows`` consecutive rows) are ranked by mean keyword hits
    per row; the top ``science_quantiles[0]`` get weight ``science_weights[0]``,
    the next ``science_quantiles[1]`` the medium weight, the rest 1.  Keys
    ``log(u)/w`` (Efraimidis–Spirakis) pick chunks until ``fraction`` of all
    train rows is reached; consecutive picked chunks form one run.
    """
    names = [n for n in scan["names"] if n.startswith("train.")]
    rows_of, cr = scan["rows"], spec.chunk_rows
    part, start, stop, mean, ordered = [], [], [], [], []
    for pi, name in enumerate(names):
        r = rows_of[name]
        starts = np.arange(0, r, cr)
        stops = np.minimum(starts + cr, r)
        sci = np.add.reduceat(scan["sci"][name].astype(np.float64), starts) if r else np.zeros(0)
        blk = block_ordered(scan["blocks"][name], spec.ordered_ratio, spec.ordered_min_adj)
        part.append(np.full(len(starts), pi)); start.append(starts); stop.append(stops)
        mean.append(sci / (stops - starts)); ordered.append(blk[starts // scan["block_rows"]])
    part, start, stop = np.concatenate(part), np.concatenate(start), np.concatenate(stop)
    mean, ordered = np.concatenate(mean), np.concatenate(ordered)
    q_hi, q_mid = spec.science_quantiles
    t_hi = float(np.quantile(mean, 1 - q_hi))
    t_mid = float(np.quantile(mean, 1 - q_hi - q_mid))
    tier = np.where(mean >= t_hi, 0, np.where(mean >= t_mid, 1, 2))
    weight = np.asarray(spec.science_weights, dtype=np.float64)[tier]
    rng = np.random.default_rng(spec.seed)
    keys = np.log(rng.random(len(part))) / weight
    order = np.argsort(-keys, kind="stable")
    total = int((stop - start).sum())
    target = int(round(spec.fraction * total))
    cum = np.cumsum((stop - start)[order])
    take = np.sort(order[:int(np.searchsorted(cum, target)) + 1])
    runs = {name: [] for name in names}
    for ci in take:
        lst = runs[names[part[ci]]]
        s, e, o = int(start[ci]), int(stop[ci]), bool(ordered[ci])
        if lst and lst[-1][1] == s and lst[-1][2] == o:
            lst[-1][1] = e                                    # extend the run
        else:
            lst.append([s, e, o])
    rows = stop - start
    stats = {"train_rows_total": total, "rows_selected": int(rows[take].sum()),
             "chunks_selected": int(len(take)), "chunks_total": int(len(part)),
             "tier_thresholds_mean_hits": {"high": round(t_hi, 4), "medium": round(t_mid, 4)},
             "tier_row_share_all": [round(float(rows[tier == t].sum() / total), 4) for t in range(3)],
             "tier_row_share_selected": [round(float(rows[take][tier[take] == t].sum() / max(rows[take].sum(), 1)), 4)
                                         for t in range(3)],
             "mean_hits_per_row_all": round(float((mean * rows).sum() / total), 4),
             "mean_hits_per_row_selected": round(float((mean[take] * rows[take]).sum() / max(rows[take].sum(), 1)), 4),
             "ordered_row_share_selected": round(float(rows[take][ordered[take]].sum() / max(rows[take].sum(), 1)), 4)}
    return {"runs": {k: v for k, v in runs.items() if v}, "stats": stats}


SIZE_KEYS = ("1", "2", "3", "4", "5", "long")
LEN_BINS = (0, 128, 256, 512, 1024, 2048, 3072, 4096, 1 << 30)


def merge_plan(cost: np.ndarray, cap: int, spec: DocMixSpec, rng: random.Random) -> list:
    """Split consecutive rows into samples: [(start, stop, size_key), ...].

    ``cost`` is the token cost of each row inside a merged sample; a sample
    never exceeds ``cap``.  The size key is drawn from ``spec.shares``; "long"
    takes n ~ U[long_min, n_fit] (n_fit = rows that still fit the cap).  When
    fewer rows fit than requested the sample is shortened; a row that alone
    exceeds the cap is skipped (counted by the caller).
    """
    keys = list(SIZE_KEYS)
    weights = [float(spec.shares.get(k, 0.0)) for k in keys]
    cs = np.concatenate([[0], np.cumsum(cost, dtype=np.int64)])
    out, i, n_rows = [], 0, len(cost)
    while i < n_rows:
        n_fit = int(np.searchsorted(cs, cs[i] + cap, side="right")) - 1 - i
        n_fit = min(n_fit, n_rows - i)
        if n_fit <= 0:
            i += 1
            continue
        key = rng.choices(keys, weights)[0]
        if key == "long":
            n = n_fit if n_fit <= spec.long_min else rng.randint(spec.long_min, n_fit)
        else:
            n = min(int(key), n_fit)
        out.append((i, i + n, key))
        i += n
    return out


def _join(rows: list, sep: str, breaks: list) -> str:
    parts = [rows[0]]
    for text, br in zip(rows[1:], breaks):
        parts.append("\n" if br else sep)
        parts.append(text)
    return "".join(parts)


def _write_arrays(prefix: str, arrays: dict) -> None:
    for suffix, arr in arrays.items():
        tmp = prefix + suffix + ".tmp"
        with open(tmp, "wb") as fh:
            arr.tofile(fh)
        os.replace(tmp, prefix + suffix)


def _len_bin(n: int) -> str:
    for lo, hi in zip(LEN_BINS, LEN_BINS[1:]):
        if n <= hi:
            return f"{lo + 1}-{hi}" if hi < (1 << 30) else f">{lo}"
    return "?"


def _sft_samples(tok, en, zh, en_len, zh_len, runs_local, spec, rng, overhead):
    """Merged chat records for one part.  ``runs_local``: [(start, stop, ordered)] in row space."""
    cap = spec.ctx_len - overhead
    prompts, replies, meta = [], [], []
    cost = np.asarray(en_len, np.int64) + np.asarray(zh_len, np.int64) + 1
    for s, e, ordered in runs_local:
        for a, b, key in merge_plan(cost[s:e], cap, spec, rng):
            a, b = a + s, b + s
            breaks = [rng.random() < spec.para_break for _ in range(b - a - 1)]
            en_text = _join(en[a:b], " ", breaks)
            zh_text = _join(zh[a:b], "", breaks)
            d = "zh_en" if rng.random() < spec.zh_en_share else "en_zh"
            src, tgt = (en_text, zh_text) if d == "en_zh" else (zh_text, en_text)
            prompts.append(f"<|im_start|>user\n{SFT_PROMPTS[d]}{src}<|im_end|>\n<|im_start|>assistant\n")
            replies.append(f"{tgt}<|im_end|>\n")
            meta.append((DIR_CODE[d], b - a, key, bool(ordered)))
    p_ids = [x.ids for x in tok.encode_batch(prompts, add_special_tokens=False)] if prompts else []
    r_ids = [x.ids for x in tok.encode_batch(replies, add_special_tokens=False)] if replies else []
    return p_ids, r_ids, meta


def _pretrain_docs(tok, en, zh, en_len, zh_len, runs_local, spec, rng, ids):
    """Merged documents for the packed pretrain stream (same merge shares)."""
    eot, en_tag, zh_tag = ids
    cost = np.asarray(en_len, np.int64) + np.asarray(zh_len, np.int64) + 1
    plan = []
    for s, e, _ordered in runs_local:
        for a, b, _key in merge_plan(cost[s:e], spec.ctx_len, spec, rng):
            breaks = [rng.random() < spec.para_break for _ in range(b - a - 1)]
            r = rng.random()
            kind = "mono" if r < spec.mono_share else (
                "zh_en" if r < spec.mono_share + (1 - spec.mono_share) * spec.zh_en_share else "en_zh")
            plan.append((kind, _join(en[a + s:b + s], " ", breaks), _join(zh[a + s:b + s], "", breaks)))
    if not plan:
        return [], {}
    enc_en = [x.ids for x in tok.encode_batch([p[1] for p in plan], add_special_tokens=False)]
    enc_zh = [x.ids for x in tok.encode_batch([p[2] for p in plan], add_special_tokens=False)]
    out, counts = [], {"mono": 0, "en_zh": 0, "zh_en": 0}
    for (kind, _e, _z), e_ids, z_ids in zip(plan, enc_en, enc_zh):
        counts[kind] += 1
        if kind == "mono":
            out.extend(z_ids); out.append(eot); out.extend(e_ids); out.append(eot)
        elif kind == "en_zh":
            out.append(en_tag); out.extend(e_ids); out.append(zh_tag); out.extend(z_ids); out.append(eot)
        else:
            out.append(zh_tag); out.extend(z_ids); out.append(en_tag); out.extend(e_ids); out.append(eot)
    return out, counts


def _build_part(args) -> dict:
    tokenizer_dir, part_path, runs, out_dir, split, spec_d, seed = args
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    spec = DocMixSpec(**spec_d)
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    dtype = _dtype_for(tok.get_vocab_size())
    special = tuple(tok.token_to_id(t) for t in ("<|endoftext|>", "<|en|>", "<|zh|>"))
    table = pq.read_table(part_path, columns=["zh", "en"])
    zh_all, en_all = table["zh"].to_pylist(), table["en"].to_pylist()
    en, zh, local = [], [], []
    for s, e, o in runs:                         # gather selected rows; runs → local spans
        local.append((len(en), len(en) + (e - s), o))
        en.extend(en_all[s:e]); zh.extend(zh_all[s:e])
    del table, zh_all, en_all
    en_len = [len(x.ids) for x in tok.encode_batch(en, add_special_tokens=False)]
    zh_len = [len(x.ids) for x in tok.encode_batch(zh, add_special_tokens=False)]
    # Chat-template overhead of one record (+8 margin: joins can re-tokenize).
    overhead = 8 + max(len(tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n",
                                      add_special_tokens=False).ids) for p in SFT_PROMPTS.values())
    stem = os.path.basename(part_path)[:-len(".parquet")]
    p_ids, r_ids, meta = _sft_samples(tok, en, zh, en_len, zh_len, local, spec, random.Random(seed), overhead)
    tokens, offsets, plens, dirs, ns = [], [0], [], [], []
    st = {"records": 0, "skipped_too_long": 0, "by_size": {}, "by_dir": {d: 0 for d in DIRECTIONS},
          "by_len": {}, "tokens_by_len": {}, "ordered_records": 0, "random_concat_records": 0,
          "sft_tokens": 0, "sft_supervised_tokens": 0, "max_len": 0}
    for p, r, (d, n, key, ordered) in zip(p_ids, r_ids, meta):
        total = len(p) + len(r)
        if total > spec.ctx_len:
            st["skipped_too_long"] += 1
            continue
        tokens.extend(p); tokens.extend(r)
        offsets.append(offsets[-1] + total); plens.append(len(p)); dirs.append(d); ns.append(min(n, 255))
        st["records"] += 1
        st["by_size"][key] = st["by_size"].get(key, 0) + 1
        st["by_dir"][DIRECTIONS[d]] += 1
        b = _len_bin(total)
        st["by_len"][b] = st["by_len"].get(b, 0) + 1
        st["tokens_by_len"][b] = st["tokens_by_len"].get(b, 0) + total
        if n > 1:
            st["ordered_records" if ordered else "random_concat_records"] += 1
        st["sft_tokens"] += total
        st["sft_supervised_tokens"] += len(r)
        st["max_len"] = max(st["max_len"], total)
    sft_dir = os.path.join(out_dir, "sft", split)
    _write_arrays(os.path.join(sft_dir, stem), {
        ".bin": np.asarray(tokens, dtype), ".idx": np.asarray(offsets, np.int64),
        ".n": np.asarray(ns, np.uint8), ".dir": np.asarray(dirs, np.uint8), ".plen": np.asarray(plens, np.int32)})
    pre, counts = _pretrain_docs(tok, en, zh, en_len, zh_len, local, spec, random.Random(seed + 1), special)
    _write_arrays(os.path.join(out_dir, "pretrain", split, stem), {".bin": np.asarray(pre, dtype)})
    return {"part": stem, "split": split, "rows": len(en), "en_tokens": int(sum(en_len)),
            "zh_tokens": int(sum(zh_len)), "pretrain_tokens": len(pre), "pretrain_docs": counts, **st}


def _merge_stats(results: list) -> dict:
    out = {}
    for r in results:
        acc = out.setdefault(r["split"], {})
        for k, v in r.items():
            if k in ("part", "split"):
                continue
            if isinstance(v, dict):
                d = acc.setdefault(k, {})
                for kk, vv in v.items():
                    d[kk] = d.get(kk, 0) + vv
            elif k == "max_len":
                acc[k] = max(acc.get(k, 0), v)
            else:
                acc[k] = acc.get(k, 0) + v
    for acc in out.values():
        recs = max(acc.get("records", 0), 1)
        acc["size_share_records"] = {k: round(acc.get("by_size", {}).get(k, 0) / recs, 4) for k in SIZE_KEYS}
        acc["dir_share_records"] = {k: round(v / recs, 4) for k, v in acc.get("by_dir", {}).items()}
        tot = max(acc.get("sft_tokens", 0), 1)
        acc["token_share_by_len"] = {k: round(v / tot, 4) for k, v in sorted(
            acc.get("tokens_by_len", {}).items(), key=lambda kv: int(kv[0].lstrip(">").split("-")[0]))}
    return out


def build_docmix(pairs_dir: str, tokenizer_dir: str, out_dir: str, spec: DocMixSpec, *, scan_dir: str,
                 workers: int = 8, log=print) -> dict:
    """Select, merge and tokenize one doc-mix subset into ``out_dir``."""
    t0 = time.time()
    scan = scan_corpus(pairs_dir, scan_dir, block_rows=spec.block_rows, workers=workers, log=log)
    sel = select_chunks(scan, spec)
    for kind in ("pretrain", "sft"):
        for split in ("train", "valid"):
            os.makedirs(os.path.join(out_dir, kind, split), exist_ok=True)
    spec_d = asdict(spec)
    jobs = []
    for i, (name, runs) in enumerate(sorted(sel["runs"].items())):
        jobs.append((tokenizer_dir, os.path.join(pairs_dir, name), runs, out_dir, "train", spec_d, spec.seed + 101 * i))
    for i, name in enumerate(n for n in scan["names"] if n.startswith("valid.")):
        rows = scan["rows"][name]
        if rows:
            ordered = bool(block_ordered(scan["blocks"][name], spec.ordered_ratio, spec.ordered_min_adj).all())
            jobs.append((tokenizer_dir, os.path.join(pairs_dir, name), [[0, rows, ordered]], out_dir, "valid",
                         spec_d, spec.seed + 7 + 101 * i))
    log(f"[docmix] {len(jobs)} parts, {sel['stats']['rows_selected']:,} train rows selected")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_build_part, jobs))
    with open(os.path.join(tokenizer_dir, "tokenizer_report.json"), encoding="utf-8") as fh:
        tok_meta = json.load(fh)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_s": round(time.time() - t0, 1), "format": "chronos-docmix-v1",
                "tokenizer_fingerprint": tok_meta.get("fingerprint"), "spec": spec_d,
                "prompts": SFT_PROMPTS, "directions": list(DIRECTIONS),
                "selection": sel["stats"], "splits": _merge_stats(results),
                "parts": sorted(results, key=lambda r: (r["split"], r["part"]))}
    with open(os.path.join(out_dir, "docmix_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    log(f"[docmix] done in {time.time() - t0:.0f}s → {out_dir}")
    return manifest


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Science-weighted, merged (n-in-1), bidirectional translation shards")
    p.add_argument("--pairs", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scan-dir", required=True, help="cache for per-row science hits and block adjacency")
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--seed", type=int, default=DocMixSpec.seed)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    spec = DocMixSpec(fraction=args.fraction, ctx_len=args.ctx, seed=args.seed)
    m = build_docmix(args.pairs, args.tokenizer, args.out, spec, scan_dir=args.scan_dir, workers=args.workers,
                     log=lambda s: print(s, flush=True))
    print(json.dumps({"selection": m["selection"], "train": {k: v for k, v in m["splits"].get("train", {}).items()
                                                               if not isinstance(v, dict) or "share" in k}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


class DocSFTRecords:
    """SFT batches that give every record the same expected passes, at any length.

    ``shards.SFTRecords`` picks the batch's target record uniformly; a batch
    holds ~budget / width records, so each short record is seen ~W_long/W_short
    times as often as a long one.  Here a record is the target with weight
    1 / rows(its length), where rows = records that one batch of that length
    holds (budget // padded width, capped at ``max_rows``): every length then
    yields records ∝ its count, i.e. the same expected passes per record for
    short sentences and 4k-token documents, floor and cap effects included.
    Reads ``.dir`` / ``.n`` and counts what was drawn per length bin
    (``seen``; stored totals in ``bin_tokens``).
    """

    def __init__(self, folder: str, dtype=np.uint16, seed: int = 0, pad_id: int = 0, tol_div: int = 16):
        stems = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".plen"))
        self.tokens, self.offsets, self.plens, rows = [], [], [], []
        for si, stem in enumerate(stems):
            base = os.path.join(folder, stem)
            off = np.fromfile(base + ".idx", dtype=np.int64)
            plen = np.fromfile(base + ".plen", dtype=np.int32)
            dirs = np.fromfile(base + ".dir", dtype=np.uint8)
            ns = np.fromfile(base + ".n", dtype=np.uint8)
            if not (len(off) == len(plen) + 1 == len(dirs) + 1 == len(ns) + 1):
                raise ValueError(f"{base}: .idx/.plen/.dir/.n lengths disagree")
            if not len(plen):
                continue
            self.tokens.append(np.memmap(base + ".bin", dtype=dtype, mode="r"))
            self.offsets.append(off); self.plens.append(plen)
            si = len(self.tokens) - 1
            rows.append(np.stack([np.full(len(plen), si), np.arange(len(plen)), np.diff(off),
                                  dirs.astype(np.int64), ns.astype(np.int64)], axis=1))
        idx = np.concatenate(rows) if rows else np.zeros((0, 5), np.int64)
        self.index = idx[np.argsort(idx[:, 2], kind="stable")]
        self.lengths = self.index[:, 2]
        self.pad_id, self.tol_div = pad_id, tol_div
        self._cum_cache: dict = {}
        self._total = int(self.lengths.sum())
        # Stored tokens per record-length bin (same bins as ``seen["len_bins"]``).
        bins = np.searchsorted(np.asarray(LEN_BINS[1:]), self.lengths, side="left")
        self.bin_tokens = {_len_bin(int(LEN_BINS[bi + 1])): int(self.lengths[bins == bi].sum())
                           for bi in np.unique(bins)}
        self.rng = np.random.default_rng(seed)
        self.seen = {"records": 0, "tokens": 0, "dir": [0, 0], "len_bins": {}}
        self.last_meta = None

    def __len__(self):
        return len(self.index)

    def _bucket(self, n, width_multiple: int = 64):
        """Padded width of a batch whose target length is ``n`` (as in ``batch``)."""
        n = np.asarray(n, dtype=np.int64)
        tol = np.maximum(4, n // self.tol_div)
        return -(-(n + tol) // width_multiple) * width_multiple

    def draw_cum(self, token_budget: int, max_rows: int = 256, width_multiple: int = 64) -> np.ndarray:
        """Cumulative target weights 1 / rows(length) for one batch shape (cached)."""
        key = (int(token_budget), int(max_rows), int(width_multiple))
        cum = self._cum_cache.get(key)
        if cum is None:
            rows = np.clip(int(token_budget) // self._bucket(self.lengths, width_multiple), 1, int(max_rows))
            cum = self._cum_cache[key] = np.cumsum(1.0 / rows)
        return cum

    def total_tokens(self) -> int:
        return self._total

    def supervised_tokens(self) -> int:
        return int(sum(int(o[-1]) - int(p.sum()) for o, p in zip(self.offsets, self.plens)))

    def batch(self, token_budget: int, max_rows: int = 256, width_multiple: int = 64):
        cum = self.draw_cum(token_budget, max_rows, width_multiple)
        i = min(int(np.searchsorted(cum, self.rng.random() * cum[-1], side="right")), len(cum) - 1)
        target = int(self.lengths[i])
        tol = max(4, target // self.tol_div)
        lo = int(np.searchsorted(self.lengths, target - tol, side="left"))
        hi = int(np.searchsorted(self.lengths, target + tol, side="right"))
        bucket = -(-(target + tol) // width_multiple) * width_multiple
        rows = max(1, min(max_rows, token_budget // max(bucket, 1), hi - lo))
        pick = self.index[lo + self.rng.choice(hi - lo, size=rows, replace=False)]
        width = -(-int(pick[:, 2].max()) // width_multiple) * width_multiple + 1
        ids = np.full((len(pick), width), self.pad_id, dtype=np.int32)
        labels = np.full((len(pick), width), -100, dtype=np.int32)
        for r, (si, ri, n, _d, _k) in enumerate(pick):
            off = self.offsets[si][ri]
            seq = self.tokens[si][off:off + n]
            ids[r, :n] = seq
            p = int(self.plens[si][ri])
            labels[r, p:n] = seq[p:]
        self.last_meta = pick[:, 2:5].copy()          # length, direction, sentences merged
        s = self.seen
        s["records"] += len(pick); s["tokens"] += int(pick[:, 2].sum())
        for d in (0, 1):
            s["dir"][d] += int((pick[:, 3] == d).sum())
        for n in pick[:, 2]:                                  # tokens drawn per record-length bin
            b = _len_bin(int(n))
            s["len_bins"][b] = s["len_bins"].get(b, 0) + int(n)
        return ids, labels
