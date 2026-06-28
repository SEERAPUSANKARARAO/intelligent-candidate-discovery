"""Tests for the ranking engine — signal monotonicity & end-to-end sanity."""

import pytest

from backend.jd_parser import parse_jd
from backend.schemas import (
    ActivitySignals,
    Candidate,
    JobRequirements,
    SignalWeights,
)
from backend import signals


def _candidate(skills, years=5.0, seniority="senior", **kw):
    base = dict(
        id="c1", name="Test", location="Remote", remote_ok=True, education="Bachelor's",
        certifications=[], available_in_weeks=0, desired_salary=100000,
        profile_completeness=0.9, role_family="Backend Engineer",
        current_title="Backend Engineer", seniority=seniority,
        total_years_experience=years, primary_domain="fintech", job_history=[],
        avg_tenure_months=24.0, job_hopping=False, role_progression=0.7,
        skills=skills, resume_summary="Experienced backend engineer.",
        activity=ActivitySignals(
            last_active_days_ago=1, applications_last_30d=5, response_rate=0.9,
            avg_response_time_hours=2.0, interview_show_rate=0.95,
            profile_update_recency_days=3),
    )
    base.update(kw)
    return Candidate(**base)


def test_skill_match_is_monotonic_in_coverage():
    req = JobRequirements(raw_text="", required_skills=["python", "sql", "aws", "docker"])
    low, _, _ = signals.skill_match(_candidate(["python"]), req)
    mid, _, _ = signals.skill_match(_candidate(["python", "sql"]), req)
    high, _, _ = signals.skill_match(_candidate(["python", "sql", "aws", "docker"]), req)
    assert low < mid < high
    assert high == pytest.approx(0.85, abs=0.16)  # full required coverage


def test_under_experience_is_penalised():
    req = JobRequirements(raw_text="", min_years_experience=8)
    under, _ = signals.experience_fit(_candidate([], years=2), req)
    meets, _ = signals.experience_fit(_candidate([], years=8), req)
    assert under < meets
    assert under < 0.5


def test_activity_score_rewards_engagement():
    active = _candidate([])
    dormant = _candidate([], activity=ActivitySignals(
        last_active_days_ago=200, applications_last_30d=0, response_rate=0.05,
        avg_response_time_hours=200.0, interview_show_rate=0.4,
        profile_update_recency_days=400))
    a_score, _ = signals.activity_score(active)
    d_score, _ = signals.activity_score(dormant)
    assert a_score > d_score
    assert a_score > 0.7 and d_score < 0.3


def test_missing_required_skills_reported():
    req = JobRequirements(raw_text="", required_skills=["python", "kafka"],
                          nice_to_have_skills=["redis"])
    _, _, extras = signals.skill_match(_candidate(["python", "redis"]), req)
    assert "kafka" in extras["missing_required_skills"]
    assert "redis" in extras["bonus_skills"]
    assert "python" in extras["matched_skills"]


# ---- end-to-end (uses real embeddings/index; slower) -------------------------------
@pytest.fixture(scope="module")
def index():
    from backend import data_gen
    from backend.ranking import CandidateIndex
    data_gen.generate()
    return CandidateIndex(data_gen.load_candidates())


def test_end_to_end_rank_returns_sorted_results(index):
    from backend.ranking import rank
    jd = ("Senior Backend Engineer, must have 5+ years, Python, microservices, "
          "PostgreSQL, distributed systems. Remote, fintech.")
    resp = rank(index, jd, weights=SignalWeights(), top_n=10)
    assert len(resp.results) == 10
    scores = [r.composite_score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
    # top result should be a backend-ish, skill-matching candidate
    top = resp.results[0]
    assert top.composite_score > 50
    assert top.rationale
    assert resp.elapsed_ms < 5000
