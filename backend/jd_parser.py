"""Deep Job Understanding — turn free-text JDs into structured requirements.

Pragmatic local NLP: taxonomy-driven skill recognition (alias-aware, so it
handles paraphrases like "ML" == "machine learning"), cue-phrase splitting of
required vs nice-to-have, and regex/keyword extraction of experience, seniority,
location and domain. The raw text is preserved for the semantic channel, which
is what ultimately captures nuance the rules miss.
"""

from __future__ import annotations

import re

from .schemas import JobRequirements
from .taxonomy import (
    SENIORITY_ALIASES,
    SENIORITY_LEVELS,
    SENIORITY_RANK,
    SKILL_ALIASES,
    find_domains,
    find_skills,
    _contains_alias,
)

# phrases that introduce a "required" vs "optional" section
_REQUIRED_CUES = ["must have", "must-have", "required", "requirements", "you have",
                  "you'll need", "essential", "we require", "minimum"]
_OPTIONAL_CUES = ["nice to have", "nice-to-have", "bonus", "a plus", "plus,", "preferred",
                  "optional", "good to have", "ideally"]

_YEARS_RE = re.compile(
    r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*\+?\s*years"   # "3-5 years"
    r"|(\d+)\s*\+?\s*years"                          # "5+ years" / "5 years"
    r"|(\d+)\s*\+?\s*yrs",
    re.IGNORECASE,
)

_TITLE_RE = re.compile(
    r"(?:hiring|seeking|looking for|need|want)\s+(?:a|an)\s+([A-Z][\w/ ]{3,40}?)"
    r"(?:\s+to\b|\s+who\b|\s+with\b|\.|,|\n)",
)

# locations we recognise (kept in sync with data_gen CITIES + regions)
_KNOWN_LOCATIONS = [
    "San Francisco", "New York", "Austin", "Seattle", "Boston", "Denver", "London",
    "Berlin", "Bangalore", "Toronto", "Amsterdam", "Singapore", "Dublin", "Chicago",
    "Europe", "US", "USA",
]


def _split_required_optional(text: str) -> tuple[str, str]:
    """Split the JD into (required-ish text, optional-ish text) by cue phrases.

    Falls back to "all required" if no optional cue is present.
    """
    lower = text.lower()
    # find earliest optional-cue position
    opt_pos = min((lower.find(c) for c in _OPTIONAL_CUES if c in lower), default=-1)
    if opt_pos == -1:
        return text, ""
    # everything from the optional cue's line onward is "optional"
    line_start = lower.rfind("\n", 0, opt_pos)
    split_at = line_start + 1 if line_start != -1 else opt_pos
    return text[:split_at], text[split_at:]


def _extract_min_years(text: str) -> float | None:
    m = _YEARS_RE.search(text)
    if not m:
        return None
    # group layout: (lo, hi) for ranges, else single in g3/g4
    if m.group(1) and m.group(2):
        return float(m.group(1))      # lower bound of a range
    for g in (m.group(3), m.group(4)):
        if g:
            return float(g)
    return None


def _extract_seniority(text: str) -> str | None:
    lower = text.lower()
    found = [lvl for lvl, aliases in SENIORITY_ALIASES.items()
             if _contains_alias(lower, aliases)]
    if not found:
        return None
    # if several, pick the most senior mentioned
    return max(found, key=lambda lvl: SENIORITY_RANK[lvl])


def _extract_location(text: str) -> str | None:
    for loc in _KNOWN_LOCATIONS:
        if loc.lower() in text.lower():
            return loc
    return None


def _extract_title(text: str) -> str | None:
    m = _TITLE_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    # fallback: first non-empty line if it looks like a title
    first = text.strip().splitlines()[0] if text.strip() else ""
    if 0 < len(first) <= 60 and "\n" not in first:
        return first.strip()
    return None


def parse_jd(jd_text: str) -> JobRequirements:
    """Parse a free-text job description into structured requirements."""
    req_text, opt_text = _split_required_optional(jd_text)

    required = find_skills(req_text)
    optional = [s for s in find_skills(opt_text) if s not in required]

    # if there was no optional section, demote a few clearly-soft skills? keep simple:
    # everything in the required block is "required", optional block is "nice to have".

    lower = jd_text.lower()
    remote = "remote" in lower or "fully remote" in lower or "remote-friendly" in lower

    domains = find_domains(jd_text)

    return JobRequirements(
        raw_text=jd_text,
        title=_extract_title(jd_text),
        required_skills=required,
        nice_to_have_skills=optional,
        min_years_experience=_extract_min_years(jd_text),
        seniority=_extract_seniority(jd_text),
        location=_extract_location(jd_text),
        remote=remote,
        domain=domains[0] if domains else None,
    )
