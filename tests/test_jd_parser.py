"""Tests for JD understanding — extraction correctness & paraphrase robustness."""

from backend.jd_parser import parse_jd


def test_extracts_skills_experience_and_remote():
    jd = (
        "We are hiring a Senior Backend Engineer. Must have 5+ years of experience, "
        "strong Python, microservices and PostgreSQL. Nice to have: Kafka and Redis. "
        "This is a remote role in fintech."
    )
    req = parse_jd(jd)
    assert req.min_years_experience == 5
    assert req.seniority == "senior"
    assert req.remote is True
    assert req.domain == "fintech"
    assert "python" in req.required_skills
    assert "microservices" in req.required_skills
    assert "postgresql" in req.required_skills
    # nice-to-have section correctly separated
    assert "kafka" in req.nice_to_have_skills
    assert "redis" in req.nice_to_have_skills
    assert "kafka" not in req.required_skills


def test_year_range_takes_lower_bound():
    req = parse_jd("Looking for an ML Engineer with 3-5 years of experience. Python required.")
    assert req.min_years_experience == 3


def test_paraphrase_maps_to_canonical_skills():
    # 'ML' and 'natural language processing' must resolve to canonical skills
    jd = "You will build ML models and work on natural language processing with large language models."
    req = parse_jd(jd)
    assert "machine learning" in req.required_skills
    assert "nlp" in req.required_skills
    assert "llm" in req.required_skills


def test_handles_no_explicit_requirements():
    req = parse_jd("A fun place to work on cool problems.")
    assert req.min_years_experience is None
    assert req.required_skills == []
    assert req.remote is False
