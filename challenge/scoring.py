"""Fusion: combine candidate features + semantic score into a final ranking score.

Thin by design — all signal computation lives in `features.py` and `honeypot.py`.
Here we (1) weight the four headline components, (2) apply the JD's explicit
disqualifier penalties (research-only, wrong-domain, framework-only), (3) apply
the behavioural multiplier and a mild location factor, and (4) floor honeypots out
of contention.
"""

from __future__ import annotations

from . import features as featmod
from . import honeypot

# fusion weights — this JD prizes product career + experience over raw skills
DEFAULT_WEIGHTS = {"semantic": 0.20, "skill": 0.25, "career": 0.33, "experience": 0.22}
# recall pre-score (no semantic): drives the shortlist, keyword-density-free
RECALL_WEIGHTS = {"skill": 0.30, "career": 0.40, "experience": 0.30}

# disqualifier penalties (JD "do NOT want") — strong down-weight, not hard zero
PENALTY = {"research_only": 0.45, "wrong_domain": 0.45, "framework_only": 0.60}


def recall_score(f: dict) -> float:
    base = (RECALL_WEIGHTS["skill"] * f["skill_score"]
            + RECALL_WEIGHTS["career"] * f["career_score"]
            + RECALL_WEIGHTS["experience"] * f["experience_score"])
    pen = 1.0
    for k, mult in PENALTY.items():
        if f.get(k):
            pen *= mult
    return base * pen


def fuse(rec: dict, f: dict, semantic: float, weights: dict | None = None,
         use_semantic: bool = True, use_behavioral: bool = True,
         use_honeypot: bool = True) -> dict:
    """Combine features + semantic into a final score + auditable breakdown."""
    w = dict(weights or DEFAULT_WEIGHTS)
    if not use_semantic:
        w["skill"] += w["semantic"] * 0.6
        w["career"] += w["semantic"] * 0.4
        w["semantic"] = 0.0

    base = (w["semantic"] * semantic + w["skill"] * f["skill_score"]
            + w["career"] * f["career_score"] + w["experience"] * f["experience_score"])

    penalty = 1.0
    for k, mult in PENALTY.items():
        if f.get(k):
            penalty *= mult
    mult = f["behavioral_multiplier"] if use_behavioral else 1.0
    location_factor = 0.85 + 0.15 * f["location_fit"]

    hp, hp_reasons = honeypot.detect(rec)
    final = base * penalty * mult * location_factor
    if use_honeypot and hp:
        final = min(final, 0.001)

    return {
        "candidate_id": rec["candidate_id"],
        "final_score": round(final, 6),
        "base_score": round(base, 4),
        "semantic": round(float(semantic), 4),
        "skill": f["skill_score"], "career": f["career_score"],
        "experience": f["experience_score"],
        "behavioral_multiplier": mult, "penalty": round(penalty, 3),
        "is_honeypot": hp, "honeypot_reasons": hp_reasons,
        "features": f,
        "reasoning": "",  # filled by reasoning.build()
    }


def structured(rec: dict) -> dict:
    """Convenience: features + recall score for one candidate (used in recall stage)."""
    f = featmod.extract(rec)
    return {"candidate_id": rec["candidate_id"], "features": f,
            "recall_score": round(recall_score(f), 6)}
