"""Per-signal scoring functions.

Each returns ``(score_0_to_1, detail_string)``. The detail string feeds the
explainability layer so every number the recruiter sees is justified. These are
the building blocks the composite reranker combines with tunable weights.
"""

from __future__ import annotations

from .schemas import Candidate, JobRequirements
from .taxonomy import SENIORITY_RANK


def skill_match(candidate: Candidate, req: JobRequirements) -> tuple[float, str, dict]:
    """Coverage of required skills (weighted high) + nice-to-have bonus.

    Returns score plus an extras dict with matched / missing / bonus skill lists
    (reused directly by the API response, no recomputation).
    """
    cand_skills = set(candidate.skills)
    required = req.required_skills
    nice = req.nice_to_have_skills

    matched_req = [s for s in required if s in cand_skills]
    missing_req = [s for s in required if s not in cand_skills]
    bonus = [s for s in nice if s in cand_skills]

    if required:
        req_cov = len(matched_req) / len(required)
    else:
        # no explicit required skills parsed — fall back to any overlap signal
        req_cov = 1.0 if (cand_skills & set(nice)) or not nice else 0.6

    nice_cov = (len(bonus) / len(nice)) if nice else 0.0
    # required coverage dominates; nice-to-have nudges up to +0.15
    score = min(1.0, 0.85 * req_cov + 0.15 * nice_cov)

    if required:
        detail = f"{len(matched_req)}/{len(required)} required skills"
        if bonus:
            detail += f", {len(bonus)}/{len(nice)} nice-to-have"
    else:
        detail = "no explicit required skills parsed; partial credit"

    extras = {
        "matched_skills": sorted(set(matched_req) | set(bonus)),
        "missing_required_skills": missing_req,
        "bonus_skills": bonus,
    }
    return score, detail, extras


def experience_fit(candidate: Candidate, req: JobRequirements) -> tuple[float, str]:
    """How well years + seniority match the ask.

    Under-experience is penalised steeply; over-qualification has mild diminishing
    returns (still a fine candidate, just maybe pricey / flight-risk).
    """
    years = candidate.total_years_experience
    min_years = req.min_years_experience

    if min_years is None:
        year_score = 0.75  # neutral-ish when JD is silent
        year_detail = f"{years:.0f} yrs (no requirement stated)"
    elif years >= min_years:
        over = years - min_years
        year_score = max(0.8, 1.0 - 0.02 * over)  # gentle decay for very over-qualified
        year_detail = f"{years:.0f} yrs meets {min_years:.0f}+ ask"
    else:
        gap = min_years - years
        year_score = max(0.0, 1.0 - 0.28 * gap)    # ~0 once ~3.5 yrs short
        year_detail = f"{years:.0f} yrs is {gap:.0f} below the {min_years:.0f}+ ask"

    # seniority alignment
    sen_score = 1.0
    sen_detail = ""
    if req.seniority and candidate.seniority:
        diff = SENIORITY_RANK[candidate.seniority] - SENIORITY_RANK[req.seniority]
        sen_score = max(0.3, 1.0 - 0.18 * abs(diff))
        if diff == 0:
            sen_detail = f"; {candidate.seniority} matches"
        elif diff < 0:
            sen_detail = f"; {candidate.seniority} below {req.seniority}"
        else:
            sen_detail = f"; {candidate.seniority} above {req.seniority}"

    score = 0.7 * year_score + 0.3 * sen_score
    return score, year_detail + sen_detail


def activity_score(candidate: Candidate) -> tuple[float, str]:
    """Behavioural / engagement signal — recency, responsiveness, reliability.

    A highly-active, responsive candidate who shows up to interviews is far more
    actionable than an equally-qualified dormant profile. This is the signal the
    challenge explicitly calls 'crucial'.
    """
    a = candidate.activity

    # recency: full credit if active in last week, decays to ~0 by ~120 days
    recency = max(0.0, 1.0 - a.last_active_days_ago / 120.0)
    # responsiveness blends reply rate and reply speed (fast = good)
    speed = max(0.0, 1.0 - a.avg_response_time_hours / 96.0)
    responsiveness = 0.6 * a.response_rate + 0.4 * speed
    # reliability
    reliability = a.interview_show_rate
    # profile freshness
    freshness = max(0.0, 1.0 - a.profile_update_recency_days / 365.0)
    # completeness
    completeness = candidate.profile_completeness

    score = (0.30 * recency + 0.30 * responsiveness + 0.20 * reliability
             + 0.10 * freshness + 0.10 * completeness)
    score = max(0.0, min(1.0, score))

    if a.last_active_days_ago <= 3 and a.response_rate >= 0.7:
        detail = f"highly active (last seen {a.last_active_days_ago}d ago, replies ~{a.avg_response_time_hours:.0f}h)"
    elif a.last_active_days_ago >= 90:
        detail = f"dormant (last active {a.last_active_days_ago}d ago)"
    else:
        detail = f"active {a.last_active_days_ago}d ago, {int(a.response_rate*100)}% response rate"
    return score, detail


def profile_fit(candidate: Candidate, req: JobRequirements) -> tuple[float, str]:
    """Location/remote compatibility, availability, domain alignment."""
    parts: list[float] = []
    notes: list[str] = []

    # location / remote
    if req.remote:
        loc_ok = candidate.remote_ok or candidate.location == "Remote"
        parts.append(1.0 if loc_ok else 0.5)
        notes.append("remote OK" if loc_ok else "not remote-flagged")
    elif req.location:
        if candidate.location.lower() == req.location.lower():
            parts.append(1.0)
            notes.append(f"in {candidate.location}")
        elif candidate.remote_ok:
            parts.append(0.8)
            notes.append("remote-flexible")
        else:
            parts.append(0.4)
            notes.append(f"in {candidate.location} (≠ {req.location})")
    else:
        parts.append(0.8)

    # availability: sooner is better
    avail = max(0.0, 1.0 - candidate.available_in_weeks / 12.0)
    parts.append(avail)
    if candidate.available_in_weeks == 0:
        notes.append("available now")
    elif candidate.available_in_weeks <= 4:
        notes.append(f"available in {candidate.available_in_weeks}w")

    # domain alignment
    if req.domain:
        dom_ok = candidate.primary_domain == req.domain
        parts.append(1.0 if dom_ok else 0.55)
        if dom_ok:
            notes.append(f"{req.domain} background")

    score = sum(parts) / len(parts)
    return score, ", ".join(notes[:3])
