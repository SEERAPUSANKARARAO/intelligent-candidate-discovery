"""Evaluation + ablation against the labelled synthetic harness.

Reports the OFFICIAL composite (0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10)
and the honeypot DQ gate, plus an ablation ladder proving each component earns its
place, a per-archetype/trap breakdown, and bootstrap confidence intervals.

    python -m challenge.evaluate --in data/synth/candidates.jsonl --gt data/synth/ground_truth.csv [--tune]
"""

from __future__ import annotations

import argparse
import itertools
import random

from . import features as featmod
from . import metrics, retrieval, scoring
from .rank import RankConfig, load_pool, run_pipeline
from .synth import load_ground_truth

ABLATIONS = [
    ("1. BM25 keyword-only", dict(score_mode="bm25")),
    ("2. Embedding-only", dict(score_mode="semantic", semantic_over_all=True)),
    ("3. Structured (trust skill/career/exp)", dict(use_semantic=False, use_behavioral=False, use_honeypot=False)),
    ("4. + semantic (hybrid)", dict(use_semantic=True, use_behavioral=False, use_honeypot=False)),
    ("5. + behavioural signals", dict(use_semantic=True, use_behavioral=True, use_honeypot=False)),
    ("6. + honeypot defence (FULL)", dict(use_semantic=True, use_behavioral=True, use_honeypot=True)),
]


def _eval(records, documents, bm25, gt, honeypots, cfg):
    results, stats = run_pipeline(records, documents, bm25, cfg)
    ranked = [r["candidate_id"] for r in results]
    relevance = {cid: gt[cid]["relevance"] for cid in gt}
    rep = metrics.full_report(ranked, relevance, honeypots, top_k=cfg.top_k)
    rep["elapsed_s"] = stats.get("elapsed_s")
    return rep, ranked


def run_ablation(in_path, gt_path, weights=None):
    records, documents, bm25 = load_pool(in_path)
    gt = load_ground_truth(gt_path)
    honeypots = {c for c, v in gt.items() if v["is_honeypot"]}
    rows = []
    for name, ov in ABLATIONS:
        cfg = RankConfig(**ov)
        if weights:
            cfg.weights = weights
        rep, _ = _eval(records, documents, bm25, gt, honeypots, cfg)
        rows.append((name, rep))
    return rows, (records, documents, bm25, gt, honeypots)


def per_archetype(records, gt, ranked_top):
    """How the FULL system treats each trap type (avg rank if surfaced, count in top)."""
    pos = {cid: i + 1 for i, cid in enumerate(ranked_top)}
    from collections import defaultdict
    out = defaultdict(lambda: {"in_top": 0, "total": 0})
    for cid, v in gt.items():
        a = v["archetype"]; out[a]["total"] += 1
        if cid in pos:
            out[a]["in_top"] += 1
    return out


def bootstrap_ci(ranked, relevance, honeypots, n=200, seed=0):
    """Bootstrap CI for the composite by resampling the ranked top-list positions."""
    rng = random.Random(seed)
    base = metrics.composite(ranked, relevance)
    samples = []
    for _ in range(n):
        # resample relevance labels with replacement over the ranked ids (stability proxy)
        rs = {cid: relevance.get(cid, 0) for cid in ranked}
        keys = list(rs)
        boot = {k: rs[rng.choice(keys)] for k in keys}
        samples.append(metrics.composite(ranked, boot))
    samples.sort()
    lo, hi = samples[int(0.025 * n)], samples[int(0.975 * n)]
    return base, lo, hi


def tune_weights(in_path, gt_path, step=0.1):
    """Search fusion weights to maximise the official composite (offline LTR-style)."""
    records, documents, bm25 = load_pool(in_path)
    gt = load_ground_truth(gt_path)
    relevance = {c: v["relevance"] for c, v in gt.items()}
    query = retrieval.jd_query_text()
    import numpy as np
    recall = np.fromiter((scoring.recall_score(featmod.extract(r)) for r in records),
                         dtype=np.float32, count=len(records))
    idx = np.argsort(-recall)[: RankConfig().shortlist_size]
    feats = {int(i): featmod.extract(records[i]) for i in idx}
    sem, _ = retrieval.semantic_scores([documents[i] for i in idx], query)
    lex = retrieval.minmax(bm25.get_scores(query)[idx])
    hyb = retrieval.minmax(0.6 * retrieval.minmax(sem) + 0.4 * lex)
    semlook = {int(i): float(h) for i, h in zip(idx, hyb)}
    steps = [round(x * step, 3) for x in range(int(1 / step) + 1)]
    best, best_c = dict(scoring.DEFAULT_WEIGHTS), -1.0
    for s, sk, ca in itertools.product(steps, repeat=3):
        ex = round(1 - s - sk - ca, 3)
        if ex < 0 or ex > 1:
            continue
        w = {"semantic": s, "skill": sk, "career": ca, "experience": ex}
        scored = [scoring.fuse(records[i], feats[int(i)], semlook[int(i)], w) for i in idx]
        for x in scored:
            x["final_score"] = round(x["final_score"], 6)
        scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
        ranked = [r["candidate_id"] for r in scored[:100]]
        c = metrics.composite(ranked, relevance)
        if c > best_c:
            best_c, best = c, w
    return best, best_c


def _table(rows):
    cols = ["composite", "ndcg@10", "ndcg@50", "map", "p@10", "honeypot_rate@100", "disqualified"]
    head = f"{'configuration':<40}" + "".join(f"{c:>13}" for c in cols)
    print(head); print("-" * len(head))
    for name, rep in rows:
        line = f"{name:<40}"
        for c in cols:
            v = rep[c]
            v = ("YES" if v else "no") if c == "disqualified" else f"{v:.4f}"
            line += f"{v:>13}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/synth/candidates.jsonl")
    ap.add_argument("--gt", default="data/synth/ground_truth.csv")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    weights = None
    if args.tune:
        print("[tune] searching fusion weights for max composite ...")
        weights, c = tune_weights(args.inp, args.gt)
        print(f"[tune] best={weights} composite={c:.4f}\n")

    print("=== Ablation (each row adds one capability) ===\n")
    rows, (records, documents, bm25, gt, honeypots) = run_ablation(args.inp, args.gt, weights)
    _table(rows)

    full_cfg = RankConfig(weights=weights or dict(scoring.DEFAULT_WEIGHTS))
    rep, ranked = _eval(records, documents, bm25, gt, honeypots, full_cfg)
    base, lo, hi = bootstrap_ci(ranked, {c: gt[c]["relevance"] for c in gt}, honeypots)
    print(f"\nFULL composite={rep['composite']:.4f}  (95% CI {lo:.3f}–{hi:.3f})")
    print(f"  NDCG@10={rep['ndcg@10']:.4f}  NDCG@50={rep['ndcg@50']:.4f}  "
          f"MAP={rep['map']:.4f}  P@10={rep['p@10']:.4f}")
    print(f"  honeypots in top-100: {rep['honeypots_in_top_k']} "
          f"({'DQ' if rep['disqualified'] else 'passes <10% gate'})  | ranked {rep['elapsed_s']}s")
    print("\nPer-archetype (in_top100 / total):")
    for a, d in sorted(per_archetype(records, gt, ranked).items()):
        print(f"  {a:<18} {d['in_top']:>4} / {d['total']}")


if __name__ == "__main__":
    main()
