"""Ranking-quality metrics — the scored quantities.

All functions take a ranked list of ``candidate_id`` (best first) and a
``relevance`` dict (candidate_id -> graded relevance, 0 if absent). This mirrors
how the organisers score: NDCG@10 is 50% of the grade, the rest split across
MRR / MAP / recall@k.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevance: Mapping[str, int], k: int = 10) -> float:
    """Normalised DCG with graded relevance (the headline metric, k=10)."""
    gains = [relevance.get(cid, 0) for cid in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: list[str], relevance: Mapping[str, int], threshold: int = 1) -> float:
    """Reciprocal rank of the first relevant (relevance >= threshold) hit."""
    for i, cid in enumerate(ranked_ids, start=1):
        if relevance.get(cid, 0) >= threshold:
            return 1.0 / i
    return 0.0


def average_precision(ranked_ids: list[str], relevance: Mapping[str, int],
                      threshold: int = 1) -> float:
    rel_total = sum(1 for v in relevance.values() if v >= threshold)
    if rel_total == 0:
        return 0.0
    hits, score = 0, 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if relevance.get(cid, 0) >= threshold:
            hits += 1
            score += hits / i
    return score / min(rel_total, len(ranked_ids)) if hits else 0.0


def recall_at_k(ranked_ids: list[str], relevance: Mapping[str, int],
                k: int = 100, threshold: int = 1) -> float:
    rel_total = sum(1 for v in relevance.values() if v >= threshold)
    if rel_total == 0:
        return 0.0
    found = sum(1 for cid in ranked_ids[:k] if relevance.get(cid, 0) >= threshold)
    return found / rel_total


def precision_at_k(ranked_ids: list[str], relevance: Mapping[str, int],
                   k: int = 10, threshold: int = 1) -> float:
    if k == 0:
        return 0.0
    return sum(1 for cid in ranked_ids[:k] if relevance.get(cid, 0) >= threshold) / k


def honeypot_rate(ranked_ids: list[str], honeypots: set[str], k: int = 100) -> float:
    """Fraction of the top-k that are honeypots. >0.10 = disqualification."""
    top = ranked_ids[:k]
    return sum(1 for cid in top if cid in honeypots) / len(top) if top else 0.0


def composite(ranked_ids: list[str], relevance: Mapping[str, int]) -> float:
    """The OFFICIAL final score:
    0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10 (relevant = tier 3+)."""
    return (0.50 * ndcg_at_k(ranked_ids, relevance, 10)
            + 0.30 * ndcg_at_k(ranked_ids, relevance, 50)
            + 0.15 * average_precision(ranked_ids, relevance, threshold=1)
            + 0.05 * precision_at_k(ranked_ids, relevance, 10, threshold=3))


def full_report(ranked_ids: list[str], relevance: Mapping[str, int],
                honeypots: set[str], top_k: int = 100) -> dict[str, float]:
    """The scoreboard, matching the official metrics + the honeypot DQ gate."""
    return {
        "composite": round(composite(ranked_ids, relevance), 4),
        "ndcg@10": round(ndcg_at_k(ranked_ids, relevance, 10), 4),
        "ndcg@50": round(ndcg_at_k(ranked_ids, relevance, 50), 4),
        "map": round(average_precision(ranked_ids, relevance, threshold=1), 4),
        "p@10": round(precision_at_k(ranked_ids, relevance, 10, threshold=3), 4),
        "p@5": round(precision_at_k(ranked_ids, relevance, 5, threshold=3), 4),
        "honeypot_rate@100": round(honeypot_rate(ranked_ids, honeypots, top_k), 4),
        "honeypots_in_top_k": sum(1 for c in ranked_ids[:top_k] if c in honeypots),
        "disqualified": honeypot_rate(ranked_ids, honeypots, top_k) > 0.10,
    }
