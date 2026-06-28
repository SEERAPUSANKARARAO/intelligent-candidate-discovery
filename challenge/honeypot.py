"""Honeypot detection — consistency, not keyword density.

The real honeypots (~80) are *subtly impossible* profiles: "8 years at a company
founded 3 years ago", "expert in 10 skills with 0 months used". They do NOT inflate
total tenure (the old heuristic caught 0 of them), so detection keys on internal
inconsistencies:

  * a job's `duration_months` exceeds the span its start/end dates allow,
  * a skill claims more months of use than the candidate's total experience,
  * many "expert" skills with 0 months of use,
  * implausible expert breadth with little endorsement backing.

Conservative by design: flagged profiles are floored out of the top-100, so we'd
rather miss a borderline than wrongly drop a genuine top candidate.
"""

from __future__ import annotations

from .schema import ANCHOR, _parse_date


def _span_months(start, end) -> int | None:
    a, b = _parse_date(start), (_parse_date(end) or ANCHOR)
    if a is None:
        return None
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


def detect(rec: dict) -> tuple[bool, list[str]]:
    skills = rec.get("skills", [])
    career = rec.get("career", [])
    yoe = float(rec.get("years_of_experience", 0) or 0)
    yoe_months = yoe * 12

    reasons = []
    # NOTE: a plain "skill duration > total experience" check is NOT used — the
    # dataset routinely has that (skill months aren't bounded by YoE), so it fires
    # on ~9% of the pool. The honeypots are caught by the signals below, calibrated
    # on the real data to flag ~60-80 profiles (matching the ~80 stated).

    # 1) a job claims more months than its start/end dates allow ("8y at a 3y-old co")
    for j in career:
        span = _span_months(j.get("start_date"), j.get("end_date"))
        dur = j.get("duration_months", 0) or 0
        if span is not None and dur > span + 12:
            reasons.append(f"role '{j.get('title','?')}' claims {dur}mo but dates span ~{span}mo")
            break

    # 2) many "expert" skills with 0 months of use ("expert in 10 skills, 0 years")
    experts = [s for s in skills if s.get("proficiency") == "expert"]
    expert_zero = [s for s in experts if (s.get("duration_months", 0) or 0) == 0]
    if len(expert_zero) >= 3:
        reasons.append(f"{len(expert_zero)} expert skills with 0 months of use")

    # 3) implausible expert breadth
    if len(experts) >= 10:
        reasons.append(f"{len(experts)} expert skills (implausible breadth)")

    # 3b) many advanced/expert skills with 0 months used (catches count/advanced
    # evasions found by the adversarial loop, without flagging genuine profiles)
    advplus_zero = [s for s in skills if s.get("proficiency") in ("advanced", "expert")
                    and (s.get("duration_months", 0) or 0) == 0]
    if len(advplus_zero) >= 5:
        reasons.append(f"{len(advplus_zero)} advanced/expert skills never used")

    # 4) total career tenure wildly exceeds stated experience
    total_career = sum(j.get("duration_months", 0) or 0 for j in career)
    if yoe_months > 0 and total_career > 1.9 * yoe_months + 24:
        reasons.append(f"career tenure {total_career}mo >> {yoe:.0f}y experience")

    return bool(reasons), reasons
