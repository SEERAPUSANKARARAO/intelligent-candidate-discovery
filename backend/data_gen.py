"""Synthetic data generation.

Produces a deterministic, internally-consistent set of candidates plus a handful
of sample job descriptions. "Internally-consistent" is the important part: a
senior candidate has more job history, longer tenure and a higher role-progression
score; an over-applied candidate has lower response quality, etc. This makes every
ranking signal genuinely informative rather than random noise.

Run directly to (re)generate the data files::

    python -m backend.data_gen
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from faker import Faker

from .schemas import ActivitySignals, Candidate, JobHistoryItem, SampleJob
from .taxonomy import (
    ALL_DOMAINS,
    CERTIFICATIONS,
    EDUCATION_LEVELS,
    ROLE_FAMILIES,
    SENIORITY_LEVELS,
    SENIORITY_YEARS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
JOBS_PATH = DATA_DIR / "jobs.json"

SEED = 42
N_CANDIDATES = 500

# weight seniority toward the middle of the ladder (realistic talent pool shape)
SENIORITY_WEIGHTS = {"intern": 6, "junior": 22, "mid": 30, "senior": 26, "lead": 11, "principal": 5}

CITIES = [
    "San Francisco", "New York", "Austin", "Seattle", "Boston", "Denver",
    "London", "Berlin", "Bangalore", "Toronto", "Remote", "Amsterdam",
    "Singapore", "Dublin", "Chicago",
]


def _summary(fake: Faker, title: str, years: float, domain: str,
             skills: list[str], seniority: str) -> str:
    """A natural-language resume blurb so the semantic channel has real prose."""
    top = ", ".join(skills[:5])
    extra = ", ".join(skills[5:8])
    templates = [
        (f"{seniority.capitalize()} {title} with {years:.0f} years building products in the "
         f"{domain} space. Deep hands-on experience with {top}. "
         f"Comfortable owning systems end to end and mentoring others. "
         f"Recent work touched {extra or 'cross-functional delivery'}."),
        (f"Results-driven {title} ({years:.0f} yrs) focused on {domain}. "
         f"Core strengths: {top}. Known for shipping reliable, well-tested software "
         f"and collaborating closely with product and design. Also worked with {extra or 'data tooling'}."),
        (f"{title} passionate about scalable, maintainable systems. "
         f"{years:.0f} years of experience, most recently in {domain}. "
         f"Toolkit includes {top}{', plus ' + extra if extra else ''}. "
         f"Strong communicator who enjoys ambiguous, high-impact problems."),
    ]
    return random.choice(templates)


def _make_job_history(fake: Faker, role_family: str, seniority: str,
                      years: float, domain: str) -> tuple[list[JobHistoryItem], float, bool, float]:
    """Generate a plausible job history and derive tenure / hopping / progression."""
    titles = ROLE_FAMILIES[role_family]["titles"]
    n_jobs = max(1, min(6, round(years / 2.5)))
    months_total = int(years * 12)
    history: list[JobHistoryItem] = []
    remaining = months_total
    for i in range(n_jobs):
        if i == n_jobs - 1:
            dur = max(4, remaining)
        else:
            # earlier roles shorter, later roles longer for "good" progression
            dur = max(4, int(remaining / (n_jobs - i)) + random.randint(-6, 8))
            dur = min(dur, remaining - (n_jobs - i - 1) * 4)
        dur = max(4, dur)
        remaining = max(0, remaining - dur)
        history.append(JobHistoryItem(
            title=random.choice(titles),
            company=fake.company(),
            domain=domain if random.random() < 0.7 else random.choice(ALL_DOMAINS),
            duration_months=dur,
        ))
    avg_tenure = sum(h.duration_months for h in history) / len(history)
    job_hopping = avg_tenure < 14 and years > 3
    # progression: more jobs + longer average tenure + higher seniority => higher
    from .taxonomy import SENIORITY_RANK
    progression = min(1.0, 0.25 + 0.12 * SENIORITY_RANK[seniority] + min(avg_tenure, 36) / 90)
    if job_hopping:
        progression *= 0.8
    return history, round(avg_tenure, 1), job_hopping, round(progression, 2)


def _make_activity(seniority: str) -> ActivitySignals:
    """Behavioural signals with realistic correlations."""
    # a fraction of candidates are 'passive' (inactive), some are 'hot' (very active)
    archetype = random.choices(["hot", "active", "passive", "dormant"],
                               weights=[18, 42, 28, 12])[0]
    if archetype == "hot":
        last_active = random.randint(0, 2)
        apps = random.randint(6, 18)
        resp_rate = round(random.uniform(0.7, 0.98), 2)
        resp_time = round(random.uniform(0.5, 6), 1)
        show = round(random.uniform(0.85, 1.0), 2)
        update_recency = random.randint(0, 10)
    elif archetype == "active":
        last_active = random.randint(1, 10)
        apps = random.randint(2, 8)
        resp_rate = round(random.uniform(0.45, 0.8), 2)
        resp_time = round(random.uniform(4, 24), 1)
        show = round(random.uniform(0.7, 0.95), 2)
        update_recency = random.randint(5, 40)
    elif archetype == "passive":
        last_active = random.randint(15, 60)
        apps = random.randint(0, 2)
        resp_rate = round(random.uniform(0.2, 0.55), 2)
        resp_time = round(random.uniform(24, 96), 1)
        show = round(random.uniform(0.55, 0.85), 2)
        update_recency = random.randint(40, 180)
    else:  # dormant
        last_active = random.randint(90, 365)
        apps = 0
        resp_rate = round(random.uniform(0.0, 0.25), 2)
        resp_time = round(random.uniform(72, 240), 1)
        show = round(random.uniform(0.3, 0.7), 2)
        update_recency = random.randint(180, 720)
    return ActivitySignals(
        last_active_days_ago=last_active,
        applications_last_30d=apps,
        response_rate=resp_rate,
        avg_response_time_hours=resp_time,
        interview_show_rate=show,
        profile_update_recency_days=update_recency,
    )


def _make_candidate(fake: Faker, idx: int) -> Candidate:
    role_family = random.choice(list(ROLE_FAMILIES.keys()))
    seniority = random.choices(SENIORITY_LEVELS,
                               weights=[SENIORITY_WEIGHTS[s] for s in SENIORITY_LEVELS])[0]
    lo, hi = SENIORITY_YEARS[seniority]
    years = round(random.uniform(lo, hi), 1)

    core = ROLE_FAMILIES[role_family]["core"]
    n_core = min(len(core), random.randint(4, 8))
    skills = random.sample(core, n_core)
    # a little cross-pollination from other families = realistic, helps semantic test
    other_pool = [s for fam, d in ROLE_FAMILIES.items() if fam != role_family for s in d["core"]]
    skills += random.sample(other_pool, random.randint(0, 3))
    skills = sorted(set(skills))

    domain = random.choice(ALL_DOMAINS)
    title = random.choice(ROLE_FAMILIES[role_family]["titles"])
    history, avg_tenure, hopping, progression = _make_job_history(
        fake, role_family, seniority, years, domain)

    education = random.choices(EDUCATION_LEVELS, weights=[12, 50, 30, 8])[0]
    certs = random.sample(CERTIFICATIONS, random.randint(0, 2))
    location = random.choice(CITIES)
    remote_ok = location == "Remote" or random.random() < 0.6

    base_salary = 60000 + int(years * 9000) + random.randint(-15000, 20000)

    completeness = round(min(1.0, 0.55
                             + 0.1 * (len(certs) > 0)
                             + 0.15 * (len(skills) >= 6)
                             + 0.1 * (len(history) >= 2)
                             + random.uniform(0, 0.15)), 2)

    return Candidate(
        id=f"cand_{idx:04d}",
        name=fake.name(),
        location=location,
        remote_ok=remote_ok,
        education=education,
        certifications=certs,
        available_in_weeks=random.choices([0, 2, 4, 8, 12], weights=[30, 25, 20, 15, 10])[0],
        desired_salary=base_salary,
        profile_completeness=completeness,
        role_family=role_family,
        current_title=title,
        seniority=seniority,
        total_years_experience=years,
        primary_domain=domain,
        job_history=history,
        avg_tenure_months=avg_tenure,
        job_hopping=hopping,
        role_progression=progression,
        skills=skills,
        resume_summary=_summary(fake, title, years, domain, skills, seniority),
        activity=_make_activity(seniority),
    )


SAMPLE_JOBS: list[SampleJob] = [
    SampleJob(
        id="job_backend_fintech",
        title="Senior Backend Engineer — Fintech",
        jd_text=(
            "We're hiring a Senior Backend Engineer to join our payments platform team at a "
            "fast-growing fintech. You will design and scale distributed systems that move money "
            "reliably.\n\n"
            "Must have: 5+ years of experience, strong Python or Go, deep knowledge of "
            "microservices and distributed systems, PostgreSQL, and REST APIs. Solid system "
            "design skills are required.\n\n"
            "Nice to have: Kafka, Redis, experience in the financial or banking domain, and "
            "exposure to Kubernetes.\n\n"
            "This is a remote-friendly role. We value engineers who communicate well and care "
            "about reliability."
        ),
    ),
    SampleJob(
        id="job_ml_genai",
        title="Machine Learning Engineer — GenAI",
        jd_text=(
            "Join our applied AI team to build LLM-powered products. We are looking for a "
            "Machine Learning Engineer with 3-5 years of experience.\n\n"
            "Required: Python, strong machine learning fundamentals, hands-on with deep learning "
            "(PyTorch or TensorFlow), and natural language processing. Experience working with "
            "large language models and generative AI is essential.\n\n"
            "Bonus: computer vision, Spark for large-scale data, and data engineering / ETL "
            "pipelines. SaaS background a plus.\n\n"
            "Hybrid role based in San Francisco."
        ),
    ),
    SampleJob(
        id="job_frontend",
        title="Frontend Engineer — E-commerce",
        jd_text=(
            "We need a Frontend Engineer to craft delightful shopping experiences for our "
            "e-commerce marketplace. 3+ years building modern web apps.\n\n"
            "Must have: TypeScript, React, strong CSS and HTML, and a great eye for UI/UX. "
            "Experience with GraphQL required.\n\n"
            "Nice to have: Vue or Angular familiarity, Node.js, and retail/e-commerce domain "
            "experience.\n\n"
            "Remote within Europe."
        ),
    ),
    SampleJob(
        id="job_devops",
        title="DevOps / Platform Engineer",
        jd_text=(
            "Looking for a Platform Engineer (DevOps) to own our cloud infrastructure and "
            "developer experience. 4+ years required.\n\n"
            "Must have: AWS, Docker, Kubernetes, Terraform, and CI/CD pipelines. Strong Linux "
            "and some Python scripting.\n\n"
            "Nice to have: GCP or Azure, experience with distributed systems at scale.\n\n"
            "Remote."
        ),
    ),
    SampleJob(
        id="job_data_eng",
        title="Data Engineer — Healthcare",
        jd_text=(
            "Senior Data Engineer to build robust data pipelines for a healthcare analytics "
            "platform. 6+ years of experience.\n\n"
            "Required: Python, SQL, Spark, Airflow, and proven data engineering / ETL experience. "
            "AWS and Kafka strongly preferred.\n\n"
            "Nice to have: healthcare or biotech domain knowledge, PostgreSQL.\n\n"
            "Based in Boston, hybrid."
        ),
    ),
    SampleJob(
        id="job_pm",
        title="Senior Product Manager",
        jd_text=(
            "We are seeking a Senior Product Manager to lead our core product. 5+ years in "
            "product management.\n\n"
            "Must have: strong product management track record, agile delivery, excellent "
            "communication and stakeholder management, and a data-informed approach (data "
            "analysis).\n\n"
            "Nice to have: leadership/mentoring experience and UI/UX sensibility. SaaS "
            "background preferred.\n\n"
            "New York, hybrid."
        ),
    ),
    SampleJob(
        id="job_paraphrase",
        title="AI Engineer (paraphrased / semantic test)",
        jd_text=(
            "We want someone who can teach machines to understand human language and build "
            "intelligent systems on top of large language models. You'll work on neural networks "
            "and ship gen AI features.\n\n"
            "We don't care about buzzwords — we care that you've built ML models end to end, are "
            "fluent in Python, and have shipped natural language products. Around 4 years of "
            "experience.\n\n"
            "Fully remote."
        ),
    ),
]


def generate(force: bool = False) -> tuple[Path, Path]:
    """Generate candidate + job data files. Idempotent unless ``force``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CANDIDATES_PATH.exists() and JOBS_PATH.exists() and not force:
        return CANDIDATES_PATH, JOBS_PATH

    random.seed(SEED)
    fake = Faker()
    Faker.seed(SEED)

    candidates = [_make_candidate(fake, i) for i in range(N_CANDIDATES)]
    CANDIDATES_PATH.write_text(
        json.dumps([c.model_dump() for c in candidates], indent=2))
    JOBS_PATH.write_text(
        json.dumps([j.model_dump() for j in SAMPLE_JOBS], indent=2))
    return CANDIDATES_PATH, JOBS_PATH


def load_candidates() -> list[Candidate]:
    if not CANDIDATES_PATH.exists():
        generate()
    raw = json.loads(CANDIDATES_PATH.read_text())
    return [Candidate(**c) for c in raw]


def load_jobs() -> list[SampleJob]:
    if not JOBS_PATH.exists():
        generate()
    raw = json.loads(JOBS_PATH.read_text())
    return [SampleJob(**j) for j in raw]


if __name__ == "__main__":
    c_path, j_path = generate(force=True)
    cands = load_candidates()
    print(f"Generated {len(cands)} candidates -> {c_path}")
    print(f"Generated {len(load_jobs())} sample jobs -> {j_path}")
