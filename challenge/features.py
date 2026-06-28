"""Single source of candidate features — used by scoring, LTR and reasoning.

Computed once per (canonical) candidate. Encodes the JD's real intent:

  * skills count only when *corroborated* by title/career (kills keyword-stuffers),
  * career evidence can credit a skill group even with no keyword (rescues the
    "plain-language Tier-5" who built a recsys but never wrote "RAG"),
  * product-company vs services/consulting (a JD disqualifier),
  * applied-ML fraction, job-hopping, research-only / wrong-domain / framework-only
    penalties (the JD's explicit "do NOT want" list),
  * behavioural availability/responsiveness from the 23 signals.

Text matching uses WORD BOUNDARIES for short/risky tokens so "ai" does not match
"retail"/"email", "rag" not "storage"/"average", "ann" not "planning". Returns
numeric features (for LTR/scoring) plus `facts`/`concerns` (for reasoning).
"""

from __future__ import annotations

import re

from . import jd

PROF_IDX = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> set[str]:
    """Tokenise lowercased text into a set of alnum words (one pass)."""
    return set(_WORD.findall(text))


def _present(term: str, toks: set[str], text: str) -> bool:
    """Single-word terms via O(1) token membership (gives word boundaries for
    free: 'ai' matches only a standalone 'ai' token, never inside 'retail').
    Multi-word / punctuated terms fall back to substring."""
    if term in toks:
        return True
    if " " in term or "-" in term or "/" in term or "." in term or "+" in term or "@" in term:
        return term in text
    return False


def _has(toks: set[str], text: str, terms: list[str]) -> bool:
    return any(_present(t.strip(), toks, text) for t in terms)


def skill_trust(sk: dict) -> float:
    """Trust a claimed skill is real (endorsements + duration + proficiency)."""
    end = min(sk.get("endorsements", 0) or 0, 50) / 50.0
    dur = min(sk.get("duration_months", 0) or 0, 48) / 48.0
    prof = PROF_IDX.get(sk.get("proficiency", "beginner"), 0) / 3.0
    trust = 0.4 * end + 0.4 * dur + 0.2 * prof
    a = sk.get("assessment_score")
    if a is not None:
        trust = min(1.0, trust + 0.15 * float(a))
    return min(1.0, trust)


def _name_match(company: str, nameset: set[str]) -> bool:
    """Match a company against a known set: multiword names by substring, short
    names by token (so 'ola' does not match 'Motorola')."""
    c = (company or "").lower()
    if not c:
        return False
    toks = set(_WORD.findall(c))
    for name in nameset:
        if " " in name or len(name) > 6:
            if name in c:
                return True
        elif name in toks:
            return True
    return False


def _classify_company(company: str, industry: str) -> str:
    ind = (industry or "").lower()
    if ind in jd.SERVICES_INDUSTRIES or _name_match(company, jd.SERVICES_COMPANIES):
        return "services"
    if ind in jd.PRODUCT_INDUSTRIES or _name_match(company, jd.PRODUCT_COMPANIES):
        return "product"
    return "other"


def extract(rec: dict) -> dict:
    skills = rec.get("skills", [])
    career = rec.get("career", [])
    sig = rec.get("signals", {})
    yoe = float(rec.get("years_of_experience", 0) or 0)

    # pre-lower skill names once (perf) and keep the originals for reasoning
    skills_lc = [(s, (s.get("name", "") or "").lower()) for s in skills]

    career_text = " ".join(j.get("description", "") for j in career).lower()
    title_text = " ".join(j.get("title", "") for j in career).lower()
    profile_text = (rec.get("summary", "") + " " + rec.get("headline", "") + " "
                    + rec.get("title", "")).lower()
    full_text = career_text + " " + profile_text + " " + title_text
    full_tok = _tok(full_text)
    career_tok = _tok(career_text)
    # tokenise each skill name once
    skills_tok = [(s, name_lc, _tok(name_lc)) for s, name_lc in skills_lc]

    # --- skill coverage: trust × corroboration (single pass, records present skills)
    matched_groups, group_strengths, present_skills = [], [], []
    seen_present = set()
    for g, forms in jd.MUST_HAVE.items():
        best_trust = 0.0
        for s, name_lc, ntok in skills_tok:
            if any(_present(f.strip(), ntok, name_lc) for f in forms):
                best_trust = max(best_trust, skill_trust(s))
                nm = s.get("name", "")
                if nm and nm not in seen_present:
                    seen_present.add(nm)
                    present_skills.append(nm)
        career_ev = 1.0 if _has(full_tok, full_text, forms) else 0.0
        strength = min(1.0, 0.6 * best_trust + 0.4 * career_ev)
        if strength > 0:
            matched_groups.append(g)
            group_strengths.append(strength)
    must_cov = sum(group_strengths) / len(jd.MUST_HAVE)
    nice_hits = sum(1 for forms in jd.NICE_TO_HAVE.values()
                    if any(any(_present(f.strip(), nt, nl) for f in forms) for _, nl, nt in skills_tok)
                    or _has(full_tok, full_text, forms))
    nice_cov = nice_hits / len(jd.NICE_TO_HAVE)
    assessment_evidence = 1.0 if any(s.get("assessment_score") for s in skills) else 0.0
    skill_score = min(1.0, 0.85 * must_cov + 0.10 * nice_cov + 0.05 * assessment_evidence)

    # --- career trajectory ---------------------------------------------------------
    title_scores, weights = [], []
    for i, j in enumerate(career):
        t = (j.get("title", "") or "").lower()
        ttok = _tok(t)
        if any(it in t or t in it for it in jd.IDEAL_TITLES):
            s = 1.0
        elif sum(_present(k, ttok, t) for k in jd.TITLE_KEYWORDS) >= 2:
            s = 0.8
        elif any(_present(k, ttok, t) for k in jd.TITLE_KEYWORDS):
            s = 0.55
        else:
            s = 0.08
        w = 1.0 if j.get("is_current") or i == 0 else 0.5 ** i
        title_scores.append(s * w)
        weights.append(w)
    title_fit = sum(title_scores) / sum(weights) if weights else 0.0

    n = max(1, len(career))
    classes = [_classify_company(j.get("company", ""), j.get("industry", "")) for j in career]
    services_frac = classes.count("services") / n
    product_frac = classes.count("product") / n
    if services_frac >= 0.9:
        services_mult = 0.30
    elif services_frac >= 0.6:
        services_mult = 0.55
    elif services_frac >= 0.3:
        services_mult = 0.80
    else:
        services_mult = 1.0

    narrative_hits = sum(1 for kw in jd.CAREER_KEYWORDS if _present(kw, full_tok, full_text))
    narrative = min(1.0, narrative_hits / 10.0)
    prod_evidence = 1.0 if _has(career_tok, career_text, jd.PRODUCTION_EVIDENCE) else 0.0
    product_bonus = min(0.12, 0.06 * classes.count("product"))
    career_base = 0.50 * title_fit + 0.32 * narrative + 0.12 * prod_evidence + product_bonus
    career_score = min(1.0, career_base * services_mult)

    # applied-ML fraction of career (token membership → 'ai' won't match 'retail')
    ml_terms = jd.TITLE_KEYWORDS + ["ml", "ai", "data scien"]
    ml_months = 0
    for j in career:
        jt = (j.get("title", "") + " " + j.get("description", "")).lower()
        if _has(_tok(jt), jt, ml_terms):
            ml_months += j.get("duration_months", 0) or 0
    total_months = sum(j.get("duration_months", 0) or 0 for j in career) or 1
    applied_ml_frac = min(1.0, ml_months / total_months)

    short = sum(1 for j in career if not j.get("is_current") and (j.get("duration_months", 0) or 0) < 18)
    avg_tenure = total_months / n
    job_hop = short >= 3 or (avg_tenure < 16 and len(career) >= 4)

    research_only = (_has(full_tok, full_text, jd.RESEARCH_ONLY_HINTS)
                     and prod_evidence == 0.0 and product_frac == 0.0)
    wrong_domain = (_has(full_tok, full_text, jd.WRONG_DOMAIN_HINTS)
                    and narrative_hits <= 2 and must_cov < 0.3)
    framework_only = (_has(full_tok, full_text, jd.FRAMEWORK_ONLY_HINTS)
                      and applied_ml_frac < 0.25 and prod_evidence == 0.0)

    # JD "plain-language Tier-5" rescue: a product-company builder whose CAREER shows
    # real retrieval/recsys/ranking work counts even with a keyword-light skills list
    # or a non-AI title. (Won't lift keyword-stuffers: they lack product + narrative +
    # production evidence; won't lift honeypots: they're floored downstream.)
    tier5_builder = (product_frac >= 0.5 and prod_evidence == 1.0 and not job_hop
                     and (narrative_hits >= 3 or applied_ml_frac >= 0.5))
    if tier5_builder:
        skill_score = max(skill_score, 0.45)
        career_score = max(career_score, 0.62)

    # --- experience + education ----------------------------------------------------
    lo, hi = jd.YOE_IDEAL
    alo, ahi = jd.YOE_ACCEPTABLE
    if lo <= yoe <= hi:
        yoe_fit = 1.0
    elif alo <= yoe <= ahi:
        yoe_fit = 0.9
    elif yoe < alo:
        yoe_fit = max(0.2, 1.0 - 0.18 * (alo - yoe))
    else:
        yoe_fit = max(0.5, 1.0 - 0.06 * (yoe - ahi))
    tiers = [e.get("tier") for e in rec.get("education", []) if e.get("tier")]
    edu_tier = min(tiers) if tiers else None
    edu_score = {1: 1.0, 2: 0.85, 3: 0.7, 4: 0.6}.get(edu_tier, 0.65)
    experience_score = min(1.0, 0.55 * yoe_fit + 0.30 * applied_ml_frac + 0.15 * edu_score)

    # --- behavioural multiplier (23 signals) ---------------------------------------
    la = sig.get("last_active_days")
    recency = 0.1 if la is None else max(0.0, 1.0 - la / 270.0)
    otw = 1.0 if sig.get("open_to_work") else 0.45
    availability = 0.6 * recency + 0.4 * otw
    rr = sig.get("recruiter_response_rate", 0.0) or 0.0
    rt = max(0.0, 1.0 - (sig.get("avg_response_time_hours", 240) or 240) / 240.0)
    ic = sig.get("interview_completion_rate", 0.0) or 0.0
    responsiveness = 0.5 * rr + 0.3 * rt + 0.2 * ic
    notice = sig.get("notice_period_days", 90)
    notice_score = max(0.2, 1.0 - notice / 120.0)
    gh = sig.get("github_activity_score", -1)
    gh = 0.0 if gh is None or gh < 0 else gh / 100.0
    oa = sig.get("offer_acceptance_rate", -1)
    oa = 0.5 if oa is None or oa < 0 else oa
    quality = (0.35 * sig.get("profile_completeness", 0.5) + 0.25 * gh + 0.20 * oa
               + 0.20 * min(1.0, sig.get("saved_by_recruiters_30d", 0) / 20.0))
    raw = 0.40 * availability + 0.30 * responsiveness + 0.15 * notice_score + 0.15 * quality
    behavioral_multiplier = round(min(1.20, max(0.50, 0.50 + raw * 0.70)), 4)
    if not sig.get("open_to_work") and (la is None or la > 120):
        behavioral_multiplier = min(behavioral_multiplier, 0.62)

    # --- location ------------------------------------------------------------------
    loc = (rec.get("location", "") + " " + rec.get("country", "")).lower()
    in_india = "india" in loc or any(c in loc for c in jd.PREFERRED_LOCATIONS)
    location_fit = 1.0 if any(c in loc for c in jd.PREFERRED_LOCATIONS) else (
        0.8 if in_india or sig.get("willing_to_relocate") else 0.45)

    concerns = []
    if services_frac >= 0.6:
        concerns.append("services/consulting-heavy career")
    if job_hop:
        concerns.append(f"short tenures (avg {avg_tenure:.0f}mo)")
    if behavioral_multiplier < 0.75:
        concerns.append("limited availability/responsiveness")
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if yoe_fit < 0.7:
        concerns.append(f"{yoe:.0f}y experience outside 5-9 band")
    if applied_ml_frac < 0.25:
        concerns.append("little hands-on applied-ML history")
    if research_only:
        concerns.append("research-leaning, thin production evidence")

    return {
        "skill_score": round(skill_score, 4), "must_cov": round(must_cov, 4),
        "nice_cov": round(nice_cov, 4), "assessment_evidence": assessment_evidence,
        "career_score": round(career_score, 4), "title_fit": round(title_fit, 4),
        "services_frac": round(services_frac, 4), "product_frac": round(product_frac, 4),
        "services_mult": services_mult, "narrative_hits": narrative_hits,
        "prod_evidence": prod_evidence, "applied_ml_frac": round(applied_ml_frac, 4),
        "avg_tenure": round(avg_tenure, 1), "job_hop": int(job_hop),
        "research_only": int(research_only), "wrong_domain": int(wrong_domain),
        "framework_only": int(framework_only),
        "experience_score": round(experience_score, 4), "yoe": yoe,
        "yoe_fit": round(yoe_fit, 4), "edu_tier": edu_tier, "edu_score": edu_score,
        "behavioral_multiplier": behavioral_multiplier,
        "availability": round(availability, 4), "responsiveness": round(responsiveness, 4),
        "notice_score": round(notice_score, 4), "location_fit": location_fit,
        "n_matched_groups": len(matched_groups), "matched_groups": matched_groups,
        "facts": {
            "title": rec.get("title", ""), "yoe": yoe, "company": rec.get("company", ""),
            "industry": rec.get("industry", ""), "present_skills": present_skills[:6],
            "n_groups": len(matched_groups), "product_frac": round(product_frac, 2),
            "response_rate": rr, "last_active_days": la, "notice": notice,
            "location": rec.get("location", ""), "applied_ml_frac": round(applied_ml_frac, 2),
        },
        "concerns": concerns,
    }


LTR_FEATURES = [
    "skill_score", "must_cov", "nice_cov", "career_score", "title_fit", "services_frac",
    "product_frac", "narrative_hits", "prod_evidence", "applied_ml_frac", "avg_tenure",
    "job_hop", "research_only", "wrong_domain", "framework_only", "experience_score",
    "yoe_fit", "edu_score", "behavioral_multiplier", "availability", "responsiveness",
    "notice_score", "location_fit", "n_matched_groups",
]
