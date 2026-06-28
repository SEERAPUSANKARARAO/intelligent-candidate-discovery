"""Fairness / bias audit for the ranking — a hiring system must be accountable.

Audits whether the top-100 over-concentrates on attributes that the JD does NOT
ask to optimise (institution prestige, location), which would be a proxy-bias risk.
Reports selection rates and disparate-impact ratios vs the pool baseline, and
offers a "blind" scoring mode that neutralises education tier + location so you can
quantify how much they move the ranking.

    python -m challenge.fairness --candidates data/demo_candidates.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter

from . import features as featmod
from . import scoring
from .rank import RankConfig, load_pool, run_pipeline


def _edu_tier(rec):
    tiers = [e.get("tier") for e in rec.get("education", []) if e.get("tier")]
    return min(tiers) if tiers else None


def _india(rec):
    loc = (rec.get("location", "") + " " + rec.get("country", "")).lower()
    return "india" in loc or "remote" in loc


def audit(records, ranked_ids) -> dict:
    by_id = {r["candidate_id"]: r for r in records}
    top = [by_id[c] for c in ranked_ids if c in by_id]
    n_pool, n_top = len(records), len(top) or 1

    def rate(pred, pop):
        return sum(1 for r in pop if pred(r)) / (len(pop) or 1)

    dims = {
        "tier1_institution": lambda r: _edu_tier(r) == 1,
        "tier1_or_2": lambda r: _edu_tier(r) in (1, 2),
        "india_located": _india,
        "product_company": lambda r: featmod._classify_company(r.get("company", ""), r.get("industry", "")) == "product",
    }
    report = {}
    for name, pred in dims.items():
        base, sel = rate(pred, records), rate(pred, top)
        di = round(sel / base, 2) if base > 0 else None  # disparate-impact ratio (top vs pool)
        report[name] = {"pool_rate": round(base, 3), "top_rate": round(sel, 3), "ratio": di}
    report["_note"] = ("Ratios far above 1 mean the attribute is over-represented in the "
                       "shortlist relative to the pool. Education tier is intentionally a "
                       "minor signal (the JD prizes product experience, not pedigree).")
    return report


def blind_rank(records, documents, bm25):
    """Re-rank with education tier + location neutralised, to quantify their effect.

    Monkeypatches the two scoring inputs for the duration of the call."""
    orig_score = featmod.extract

    def blind_extract(rec):
        f = orig_score(rec)
        f["edu_score"] = 0.65
        f["experience_score"] = round(min(1.0, 0.55 * f["yoe_fit"] + 0.30 * f["applied_ml_frac"]
                                          + 0.15 * 0.65), 4)
        f["location_fit"] = 0.8
        return f

    featmod.extract = blind_extract
    try:
        res, _ = run_pipeline(records, documents, bm25, RankConfig())
        return [r["candidate_id"] for r in res]
    finally:
        featmod.extract = orig_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/demo_candidates.jsonl")
    args = ap.parse_args()
    records, documents, bm25 = load_pool(args.candidates)
    res, _ = run_pipeline(records, documents, bm25, RankConfig())
    ranked = [r["candidate_id"] for r in res]

    print("=== Fairness audit (top-100 vs pool) ===")
    rep = audit(records, ranked)
    for k, v in rep.items():
        if k.startswith("_"):
            continue
        print(f"  {k:<20} pool={v['pool_rate']:.2f}  top={v['top_rate']:.2f}  ratio={v['ratio']}")
    print("\n" + rep["_note"])

    blind = blind_rank(records, documents, bm25)
    overlap = len(set(ranked[:50]) & set(blind[:50]))
    print(f"\nBlind mode (no edu tier / location): top-50 overlap with default = "
          f"{overlap}/50 ({overlap*2}%) — high overlap ⇒ ranking is driven by merit, "
          f"not pedigree/location.")


if __name__ == "__main__":
    main()
