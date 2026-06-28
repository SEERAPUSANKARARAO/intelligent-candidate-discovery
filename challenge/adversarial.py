"""Adversarial honeypot red-team / blue-team loop.

Red-team synthesises honeypots designed to EVADE our detector across several
strategies; blue-team (challenge.honeypot.detect) tries to catch them. We report
catch-rate per strategy so we know exactly where the detector is weak — rather than
trusting it against only the traps we happened to plant. Surviving strategies are
candidates for hardening.

    python -m challenge.adversarial --rounds 3 --per 200
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta

from . import honeypot

_AI = ["Embeddings", "FAISS", "BM25", "RAG", "LLMs", "Transformers", "Vector Search",
       "Reranking", "NDCG", "Recommendation Systems", "PyTorch", "BERT", "Qdrant"]


def _iso(d):
    return d.strftime("%Y-%m-%d")


def _make(strategy: str, rng: random.Random) -> dict:
    """Build a fabricated profile per evasion strategy."""
    yoe = rng.uniform(2, 6)
    end = date(2026, 6, 1)
    n_expert0 = {"date_evade": 0, "count_evade": 2, "few_expert": 1,
                 "naive": 6, "mixed": 2}.get(strategy, 2)
    n_expert_used = {"count_evade": 7, "naive": 0}.get(strategy, 1)
    skills = []
    for nm in rng.sample(_AI, min(len(_AI), n_expert0 + n_expert_used + rng.randint(0, 2))):
        if n_expert0 > 0:
            skills.append({"name": nm, "proficiency": "expert", "endorsements": rng.randint(0, 2),
                           "duration_months": 0}); n_expert0 -= 1
        elif n_expert_used > 0:
            skills.append({"name": nm, "proficiency": "expert", "endorsements": rng.randint(5, 30),
                           "duration_months": rng.randint(12, 40)}); n_expert_used -= 1
        else:
            skills.append({"name": nm, "proficiency": "advanced", "endorsements": rng.randint(0, 5),
                           "duration_months": rng.randint(0, 6)})
    # career: date_evade keeps duration consistent with the span; others may overclaim
    months = int(yoe * 12)
    if strategy in ("date_evade", "count_evade", "few_expert"):
        span = months  # consistent → evades the date check
    else:
        span = max(6, int(months / rng.uniform(2.6, 4.0)))  # impossible (naive/mixed)
    start = end - timedelta(days=int(span * 30.4))
    career = [{"title": "AI Engineer", "company": "Globex", "industry": "AI/ML",
               "start_date": _iso(start), "end_date": None, "duration_months": months,
               "is_current": True, "description": "Expert in embeddings, FAISS, RAG, NDCG."}]
    return {"candidate_id": "ADV", "years_of_experience": round(yoe, 1),
            "skills": skills, "career": career}


def run(rounds=3, per=200, seed=0) -> dict:
    rng = random.Random(seed)
    strategies = ["naive", "date_evade", "count_evade", "few_expert", "mixed"]
    report = {}
    for strat in strategies:
        caught = 0
        for _ in range(per * rounds):
            rec = _make(strat, rng)
            if honeypot.detect(rec)[0]:
                caught += 1
        total = per * rounds
        report[strat] = {"caught": caught, "total": total, "catch_rate": round(caught / total, 3)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per", type=int, default=200)
    args = ap.parse_args()
    rep = run(args.rounds, args.per)
    print("=== Adversarial honeypot catch-rate by evasion strategy ===")
    for strat, d in rep.items():
        flag = "" if d["catch_rate"] >= 0.9 else "   <-- WEAK (hardening candidate)"
        print(f"  {strat:<12} {d['catch_rate']*100:5.1f}%  ({d['caught']}/{d['total']}){flag}")
    weak = [s for s, d in rep.items() if d["catch_rate"] < 0.9]
    print("\nSurviving strategies:", weak or "none — detector robust across tested evasions")


if __name__ == "__main__":
    main()
