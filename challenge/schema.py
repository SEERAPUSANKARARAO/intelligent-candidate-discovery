"""Real Redrob schema → canonical record adapter (+ safe streaming loader).

The official `candidates.jsonl` nests fields (`profile.*`, `redrob_signals.*`),
uses date strings and `-1` sentinels, and stores per-skill assessment scores in a
separate dict. The rest of the engine consumes a single flat **canonical** record
so scoring/retrieval/reasoning never touch the raw shape. This is the one place
that knows the real schema — if the organisers tweak a field, only this file
changes.

Robustness: every access is defensive (`.get`), malformed JSON lines are skipped,
and `.jsonl` / `.jsonl.gz` are both supported.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

# Recency anchor: days-since-active is measured relative to this date. Fixed so the
# pipeline is deterministic and single-pass (relative ordering is what matters).
ANCHOR = date(2026, 6, 27)

_TIER_MAP = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": None}


def _parse_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        y, m, d = (int(x) for x in s[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


def _days_since(s, anchor: date = ANCHOR):
    d = _parse_date(s)
    if d is None:
        return None
    return max(0, (anchor - d).days)


def adapt(rec: dict, anchor: date = ANCHOR) -> dict:
    """Map one raw candidate record to the canonical flat form used everywhere."""
    p = rec.get("profile", {}) or {}
    sig = rec.get("redrob_signals", {}) or {}
    assess = sig.get("skill_assessment_scores", {}) or {}
    assess_lc = {str(k).lower(): v for k, v in assess.items()} if isinstance(assess, dict) else {}

    # skills: attach normalised assessment score (0-1) when the platform tested it
    skills = []
    for s in rec.get("skills", []) or []:
        if not isinstance(s, dict):
            continue
        name = s.get("name", "")
        a = assess_lc.get(str(name).lower())
        skills.append({
            "name": name,
            "proficiency": s.get("proficiency", "beginner"),
            "endorsements": s.get("endorsements", 0) or 0,
            "duration_months": s.get("duration_months", 0) or 0,
            "assessment_score": (float(a) / 100.0) if isinstance(a, (int, float)) else None,
        })

    career = []
    for j in rec.get("career_history", []) or []:
        if not isinstance(j, dict):
            continue
        career.append({
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "industry": j.get("industry", ""),
            "company_size": j.get("company_size", ""),
            "start_date": j.get("start_date"),
            "end_date": j.get("end_date"),
            "duration_months": j.get("duration_months", 0) or 0,
            "is_current": bool(j.get("is_current", False)),
            "description": j.get("description", "") or "",
        })

    education = []
    for e in rec.get("education", []) or []:
        if not isinstance(e, dict):
            continue
        education.append({
            "institution": e.get("institution", ""),
            "degree": e.get("degree", ""),
            "field": e.get("field_of_study", ""),
            "tier": _TIER_MAP.get(e.get("tier"), None),
            "start_year": e.get("start_year"),
            "end_year": e.get("end_year"),
        })

    salary = sig.get("expected_salary_range_inr_lpa", {}) or {}
    signals = {
        "profile_completeness": (sig.get("profile_completeness_score", 0) or 0) / 100.0,
        "last_active_days": _days_since(sig.get("last_active_date"), anchor),
        "signup_days": _days_since(sig.get("signup_date"), anchor),
        "open_to_work": bool(sig.get("open_to_work_flag", False)),
        "recruiter_response_rate": sig.get("recruiter_response_rate", 0.0) or 0.0,
        "avg_response_time_hours": sig.get("avg_response_time_hours", 240.0),
        "interview_completion_rate": sig.get("interview_completion_rate", 0.0) or 0.0,
        "offer_acceptance_rate": sig.get("offer_acceptance_rate", -1),
        "notice_period_days": sig.get("notice_period_days", 90),
        "github_activity_score": sig.get("github_activity_score", -1),
        "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d", 0) or 0,
        "search_appearance_30d": sig.get("search_appearance_30d", 0) or 0,
        "profile_views_30d": sig.get("profile_views_received_30d", 0) or 0,
        "applications_30d": sig.get("applications_submitted_30d", 0) or 0,
        "connection_count": sig.get("connection_count", 0) or 0,
        "endorsements_received": sig.get("endorsements_received", 0) or 0,
        "willing_to_relocate": bool(sig.get("willing_to_relocate", False)),
        "preferred_work_mode": sig.get("preferred_work_mode", ""),
        "verified_email": bool(sig.get("verified_email", False)),
        "verified_phone": bool(sig.get("verified_phone", False)),
        "linkedin_connected": bool(sig.get("linkedin_connected", False)),
        "salary_min": salary.get("min"),
        "salary_max": salary.get("max"),
    }

    return {
        "candidate_id": rec.get("candidate_id", ""),
        "name": p.get("anonymized_name", ""),
        "headline": p.get("headline", ""),
        "summary": p.get("summary", "") or "",
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "years_of_experience": float(p.get("years_of_experience", 0) or 0),
        "title": p.get("current_title", ""),
        "company": p.get("current_company", ""),
        "company_size": p.get("current_company_size", ""),
        "industry": p.get("current_industry", ""),
        "skills": skills,
        "career": career,
        "education": education,
        "signals": signals,
    }


def iter_raw(path: str | Path):
    """Stream raw records from .jsonl or .jsonl.gz, skipping malformed lines."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue  # never let one bad row kill the run


def load_candidates(path: str | Path, anchor: date = ANCHOR) -> list[dict]:
    """Load + adapt every candidate to canonical form (memory-flat streaming)."""
    return [adapt(r, anchor) for r in iter_raw(path)]
