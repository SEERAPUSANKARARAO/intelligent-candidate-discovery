"""Pydantic models shared across the API, ranking engine and data layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobHistoryItem(BaseModel):
    title: str
    company: str
    domain: str
    duration_months: int


class ActivitySignals(BaseModel):
    """Crucial behavioural / engagement signals the challenge calls out."""

    last_active_days_ago: int
    applications_last_30d: int
    response_rate: float = Field(ge=0, le=1)        # replies to recruiter outreach
    avg_response_time_hours: float
    interview_show_rate: float = Field(ge=0, le=1)
    profile_update_recency_days: int


class Candidate(BaseModel):
    id: str
    name: str
    # --- profile attributes ---
    location: str
    remote_ok: bool
    education: str
    certifications: list[str] = []
    available_in_weeks: int          # 0 == immediate
    desired_salary: int
    profile_completeness: float = Field(ge=0, le=1)
    # --- career metadata ---
    role_family: str
    current_title: str
    seniority: str
    total_years_experience: float
    primary_domain: str
    job_history: list[JobHistoryItem] = []
    avg_tenure_months: float
    job_hopping: bool
    role_progression: float = Field(ge=0, le=1)
    # --- skills + prose ---
    skills: list[str]
    resume_summary: str
    # --- activity ---
    activity: ActivitySignals


class JobRequirements(BaseModel):
    """Structured output of the JD parser — the system's understanding of a JD."""

    raw_text: str
    title: str | None = None
    required_skills: list[str] = []
    nice_to_have_skills: list[str] = []
    min_years_experience: float | None = None
    seniority: str | None = None
    location: str | None = None
    remote: bool = False
    domain: str | None = None


class SignalWeights(BaseModel):
    """Tunable weights for the composite rerank. Need not sum to 1 — normalised internally."""

    semantic_fit: float = 0.30
    skill_match: float = 0.30
    experience_fit: float = 0.15
    activity: float = 0.15
    profile_fit: float = 0.10


class ScoreComponent(BaseModel):
    name: str
    score: float          # 0..1 raw sub-score
    weight: float         # normalised weight applied
    contribution: float   # score * weight (points toward composite, 0..1)
    detail: str = ""


class RankedResult(BaseModel):
    candidate: Candidate
    rank: int
    composite_score: float                 # 0..100 for display
    components: list[ScoreComponent]
    matched_skills: list[str]
    missing_required_skills: list[str]
    bonus_skills: list[str]
    feedback_adjustment: float = 0.0       # points added/removed by feedback loop
    rationale: str


class RankRequest(BaseModel):
    jd_text: str
    weights: SignalWeights | None = None
    top_n: int = 10
    job_id: str | None = None              # ties feedback to a specific JD


class RankResponse(BaseModel):
    requirements: JobRequirements
    results: list[RankedResult]
    total_candidates: int
    retrieved: int
    elapsed_ms: float
    weights: SignalWeights


class FeedbackRequest(BaseModel):
    job_id: str
    candidate_id: str
    vote: int = Field(description="+1 thumbs up, -1 thumbs down")


class SampleJob(BaseModel):
    id: str
    title: str
    jd_text: str
