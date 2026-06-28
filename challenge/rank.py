"""The ranking funnel — candidates.jsonl → submission.csv (official format).

Stages (all offline, CPU, well inside 5 min / 16 GB):
  1. Load + adapt to canonical schema; build BM25 over the pool.
  2. Structured pre-score over ALL candidates (trust + career + experience −
     disqualifier penalties). Keyword density never drives recall, so honeypots
     and keyword-stuffers cannot crowd the shortlist.
  3. Shortlist top-N by recall score; hybrid (BM25 ⊕ semantic) rerank on it.
  4. Fuse → behavioural multiplier → honeypot floor → top-100 with reasoning.

Output CSV columns: candidate_id,rank,score,reasoning (exact spec order), score
non-increasing, ties broken by candidate_id ascending (official validator rule).

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as featmod
from . import jd, reasoning, retrieval, scoring
from .schema import load_candidates

# Optional-depth modules (embed_index/ltr/rerank) are imported LAZILY inside the
# helpers below so the DEFAULT pipeline needs only numpy/scipy/scikit-learn
# (ltr → synth → faker etc. must NOT load for a plain `python rank.py`).
_emb_cache: dict = {}
_ltr_cache: dict = {}


def _get_emb(art_dir):
    if art_dir not in _emb_cache:
        try:
            from .embed_index import PrecomputedEmbeddings
            _emb_cache[art_dir] = PrecomputedEmbeddings(art_dir)
        except Exception:
            _emb_cache[art_dir] = None
    return _emb_cache[art_dir]


def _get_ltr(path):
    key = path or "default"
    if key not in _ltr_cache:
        try:
            from . import ltr as ltrmod
            _ltr_cache[key] = ltrmod.LTRModel(path or str(ltrmod.ARTIFACT))
        except Exception:
            _ltr_cache[key] = None
    return _ltr_cache[key]


@dataclass
class RankConfig:
    shortlist_size: int = 2000
    top_k: int = jd.TOP_K
    weights: dict = field(default_factory=lambda: dict(scoring.DEFAULT_WEIGHTS))
    use_semantic: bool = True
    use_behavioral: bool = True
    use_honeypot: bool = True
    score_mode: str = "fusion"          # fusion | bm25 | semantic (baselines)
    semantic_over_all: bool = False
    embeddings_dir: str | None = None   # precomputed BGE artifacts (opt-in)
    use_ltr: bool = False               # ensemble with LambdaMART (opt-in)
    ltr_path: str | None = None
    ltr_weight: float = 0.4
    use_rerank: bool = False            # cross-encoder rerank top-N (opt-in)
    rerank_n: int = 120


def load_pool(path: str | Path):
    """Load canonical candidates + searchable documents + BM25 index."""
    records = load_candidates(path)
    documents = [retrieval.candidate_document(r) for r in records]
    bm25 = retrieval.BM25Index(documents)
    return records, documents, bm25


def run_pipeline(records, documents, bm25, cfg: RankConfig):
    t0 = time.perf_counter()
    query = retrieval.jd_query_text()
    bm25_scores = bm25.get_scores(query)

    # --- single-channel baselines (for the ablation) ----------------------------
    if cfg.score_mode == "bm25":
        order = np.argsort(-bm25_scores)[: cfg.top_k]
        res = [{"candidate_id": records[i]["candidate_id"], "final_score": float(bm25_scores[i]),
                "reasoning": "BM25 lexical match (baseline)."} for i in order]
        return _finalize(res, cfg.top_k), {"backend": "bm25",
                                           "elapsed_s": round(time.perf_counter() - t0, 2)}
    if cfg.score_mode == "semantic":
        idx = (np.arange(len(records)) if cfg.semantic_over_all
               else np.argsort(-bm25_scores)[: cfg.shortlist_size])
        sem, backend = retrieval.semantic_scores([documents[i] for i in idx], query)
        full = np.full(len(records), -1.0, dtype=np.float32)
        full[idx] = sem
        order = np.argsort(-full)[: cfg.top_k]
        res = [{"candidate_id": records[i]["candidate_id"], "final_score": float(full[i]),
                "reasoning": f"Semantic match ({backend}, baseline)."} for i in order]
        return _finalize(res, cfg.top_k), {"backend": backend,
                                           "elapsed_s": round(time.perf_counter() - t0, 2)}

    # --- fusion path ------------------------------------------------------------
    # Stage 2: structured pre-score over ALL candidates. Compute only the recall
    # float per candidate and discard the feature dict immediately, so we never
    # hold 100k feature dicts in memory (just one float32 array).
    recall = np.fromiter((scoring.recall_score(featmod.extract(r)) for r in records),
                         dtype=np.float32, count=len(records))
    n_short = min(cfg.shortlist_size, len(records))
    shortlist_idx = np.argsort(-recall)[:n_short]

    # Stage 3: semantic over shortlist — precomputed BGE if provided, else TF-IDF —
    # blended with BM25 (hybrid). fuse() handles weight redistribution if disabled.
    backend = "disabled"
    sem_lookup = {}
    if cfg.use_semantic:
        pe = _get_emb(cfg.embeddings_dir) if cfg.embeddings_dir else None
        if pe is not None:
            sem_short = np.array([pe.score_for(records[i]["candidate_id"]) for i in shortlist_idx],
                                 dtype=np.float32)
            backend = "bge-precomputed"
        else:
            sem_short, backend = retrieval.semantic_scores([documents[i] for i in shortlist_idx], query)
        sem_short = retrieval.minmax(sem_short)
        lex_short = retrieval.minmax(bm25_scores[shortlist_idx])
        hybrid = retrieval.minmax(0.6 * sem_short + 0.4 * lex_short)
        sem_lookup = {int(i): float(h) for i, h in zip(shortlist_idx, hybrid)}
        backend = f"hybrid(bm25 + {backend})"

    # Stage 4: features + fuse.
    scored, feats = [], []
    for i in shortlist_idx:
        f = featmod.extract(records[i])
        feats.append(f)
        scored.append(scoring.fuse(records[i], f, sem_lookup.get(int(i), 0.0),
                      cfg.weights, cfg.use_semantic, cfg.use_behavioral, cfg.use_honeypot))

    # Optional LambdaMART ensemble (honeypots stay floored).
    if cfg.use_ltr:
        model = _get_ltr(cfg.ltr_path)
        if model is not None:
            ltr_n = retrieval.minmax(np.asarray(model.score_features(feats), dtype=np.float32))
            fus_n = retrieval.minmax(np.asarray([s["final_score"] for s in scored], dtype=np.float32))
            for s, fn, ln in zip(scored, fus_n, ltr_n):
                if not s["is_honeypot"]:
                    s["final_score"] = float((1 - cfg.ltr_weight) * fn + cfg.ltr_weight * ln)
            backend += " +ltr"

    # Optional cross-encoder rerank of the top-N.
    if cfg.use_rerank:
        from . import rerank as rerankmod
        if rerankmod.available():
            scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
            texts = {records[i]["candidate_id"]: documents[i] for i in shortlist_idx}
            scored = rerankmod.rerank(scored, texts, cfg.rerank_n)
            backend += " +cross-encoder"

    results = _finalize(scored, cfg.top_k, with_reasoning=True)
    stats = {"backend": backend, "shortlist": int(n_short),
             "elapsed_s": round(time.perf_counter() - t0, 2),
             "honeypots_in_top_k": sum(1 for r in results if r.get("is_honeypot"))}
    return results, stats


def _finalize(scored: list[dict], top_k: int, with_reasoning: bool = False) -> list[dict]:
    """Round, sort by (-score, candidate_id) [official tie rule], take top-K, rank."""
    # Normalise to [0,1] FIRST, then round, THEN sort by (-score, candidate_id).
    # Doing it in this order guarantees the official tie rule (equal emitted scores
    # ⇒ candidate_id ascending) holds on the exact values we write.
    top_score = max((float(s["final_score"]) for s in scored), default=1.0) or 1.0
    for s in scored:
        s["final_score"] = round(float(s["final_score"]) / top_score, 6)
    scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    top = scored[: min(top_k, len(scored))]
    for rank, s in enumerate(top, start=1):
        s["rank"] = rank
        if with_reasoning:
            s["reasoning"] = reasoning.build(s, rank, top_k)
    return top


def write_submission(results: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])  # exact spec order
        for r in results:
            w.writerow([r["candidate_id"], r["rank"], f"{r['final_score']:.6f}",
                        r.get("reasoning", "")])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank candidates → submission.csv")
    ap.add_argument("--candidates", "--in", dest="candidates", default="data/candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--shortlist", type=int, default=2000)
    ap.add_argument("--embeddings", default=None, help="precomputed BGE artifacts dir")
    ap.add_argument("--ltr", action="store_true", help="ensemble with LambdaMART")
    ap.add_argument("--rerank", action="store_true", help="cross-encoder rerank top-N")
    args = ap.parse_args()

    print(f"[rank] loading + indexing {args.candidates} ...")
    t0 = time.perf_counter()
    records, documents, bm25 = load_pool(args.candidates)
    print(f"[rank] indexed {len(records):,} candidates in {time.perf_counter()-t0:.1f}s")

    results, stats = run_pipeline(records, documents, bm25,
                                  RankConfig(shortlist_size=args.shortlist,
                                             embeddings_dir=args.embeddings,
                                             use_ltr=args.ltr, use_rerank=args.rerank))
    out = write_submission(results, args.out)
    print(f"[rank] backend={stats.get('backend')} | ranked in {stats['elapsed_s']}s | "
          f"honeypots in top-100: {stats.get('honeypots_in_top_k', 0)}")
    print(f"[rank] wrote {out}\n[rank] top 5:")
    for r in results[:5]:
        print(f"  {r['rank']:>2}. {r['candidate_id']} {r['final_score']:.4f}  {r['reasoning'][:95]}")


if __name__ == "__main__":
    main()
