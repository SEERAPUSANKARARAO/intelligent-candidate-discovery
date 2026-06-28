"""Synthetic data generator — REAL schema, graded tiers, and the four traps.

Emits records in the official nested schema (`profile`, `career_history`,
`education`, `skills`, `redrob_signals`, CAND_XXXXXXX ids) plus a sidecar
`ground_truth.csv` (candidate_id -> relevance tier 0-5, is_honeypot, archetype).

Why: the real ground truth is hidden and we get only 3 submissions, so we need a
labelled local harness to measure the official composite (NDCG@10/@50, MAP, P@10)
and to prove trap resistance. Reproduces the four documented traps:

  * keyword-stuffer  — non-AI title + many AI skills            -> tier 0/1
  * plain Tier-5     — rich AI career text, keyword-light skills -> tier 4/5
  * behavioral-twin  — identical profile, active vs dormant      -> tiers split
  * honeypot         — date/duration impossibility, expert-0-mo  -> tier 0

    python -m challenge.synth --n 5000 --out data/synth
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

# representative real surface skills per JD must-have group
GROUP_SKILLS = {
    "embeddings": ["Sentence Transformers", "BGE Embeddings", "E5", "Dense Retrieval"],
    "vector_db": ["FAISS", "Qdrant", "Milvus", "Pinecone", "Elasticsearch"],
    "ranking_ir": ["BM25", "Learning to Rank", "Cross-Encoder Reranking", "Hybrid Search"],
    "evaluation": ["NDCG", "MRR", "A/B Testing", "Offline Evaluation"],
    "python": ["Python", "PyTorch", "NumPy"],
    "nlp_llm": ["Transformers", "BERT", "RAG", "LLMs", "Semantic Search"],
    "ml_core": ["Deep Learning", "Machine Learning", "TensorFlow"],
    "search_rec": ["Recommendation Systems", "Vector Search", "Personalization"],
}
NICE_SKILLS = ["LoRA", "QLoRA", "PEFT", "LambdaMART", "XGBoost", "MLflow", "Kubernetes",
               "Spark", "Airflow", "Kafka"]
OFF = {
    "Marketing Manager": ["SEO", "Google Ads", "Content Strategy", "HubSpot", "Branding"],
    "HR Manager": ["Recruiting", "Onboarding", "Payroll", "HRIS", "Employee Relations"],
    "Accountant": ["Excel", "Tally", "GST", "Auditing", "QuickBooks"],
    "Mechanical Engineer": ["AutoCAD", "SolidWorks", "Thermodynamics", "CATIA"],
    "Sales Executive": ["CRM", "Lead Generation", "Negotiation", "Salesforce"],
    "Content Writer": ["Copywriting", "WordPress", "SEO", "Editing"],
    "Civil Engineer": ["AutoCAD", "STAAD Pro", "Project Estimation"],
    "Backend Engineer": ["Java", "Spring Boot", "PostgreSQL", "REST APIs", "Microservices"],
}
AI_TITLES = ["Machine Learning Engineer", "Senior Machine Learning Engineer", "AI Engineer",
             "NLP Engineer", "Search Engineer", "Applied Scientist",
             "Recommendation Systems Engineer", "Senior AI Engineer"]
PRODUCT_COS = ["Flipkart", "Zomato", "Swiggy", "CRED", "Razorpay", "Sarvam AI", "Meesho",
               "PhonePe", "Sprinklr", "Google", "Microsoft", "Netflix", "Uber"]
SERVICES_COS = ["TCS", "Infosys", "Wipro", "Cognizant", "Accenture", "Capgemini", "Mindtree"]
OTHER_COS = ["Globex", "Initech", "Umbrella Corp", "Hooli", "Soylent"]
PROD_DESC = ("Built and shipped a {x} system in production serving millions of users; "
             "owned embeddings, hybrid retrieval and reranking; improved NDCG and reduced "
             "latency via FAISS ANN indexing and A/B tested ranking changes.")
PLAIN_DESC = ("Led the team that designed our search and recommendation engine end to end — "
              "we moved from keyword lookup to a vector-based matching system, ran offline "
              "and online evaluations, and scaled it to production traffic.")
OFF_DESC = "Handled {x} responsibilities, coordinated with stakeholders, and delivered on targets."
TIERS = ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]
INSTITUTIONS = ["IIT Bombay", "IIT Delhi", "IISc Bangalore", "BITS Pilani", "NIT Trichy",
                "VIT Vellore", "Anna University", "State University"]
LOCS = ["Bangalore, Karnataka", "Pune, Maharashtra", "Hyderabad, Telangana", "Noida, UP",
        "Delhi NCR", "Chennai, Tamil Nadu", "Mumbai, Maharashtra", "Remote"]
SIZES = ["11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"]

# archetype -> (weight, relevance tier)
ARCHE = {
    "perfect": (0.5, 5), "strong": (2.5, 4), "plain_tier5": (0.6, 5), "good": (5.0, 3),
    "moderate": (10.0, 2), "weak": (20.0, 1), "irrelevant": (56.0, 0),
    "keyword_stuffer": (3.0, 0), "honeypot": (1.9, 0),
}


def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _job(rng, title, company, industry, months, end: date, desc, honeypot=False):
    if honeypot:  # claim more months than the date span allows
        span = max(6, int(months / rng.uniform(2.6, 4.0)))
        start = end - timedelta(days=int(span * 30.4))
    else:
        start = end - timedelta(days=int(months * 30.4))
    return {
        "company": company, "title": title, "start_date": _iso(start),
        "end_date": None if end >= date(2026, 6, 1) else _iso(end),
        "duration_months": months, "is_current": end >= date(2026, 6, 1),
        "industry": industry, "company_size": rng.choice(SIZES), "description": desc,
    }


def _skill(rng, name, genuine):
    if genuine:
        prof = rng.choices(["intermediate", "advanced", "expert"], [3, 4, 3])[0]
        dur = rng.randint(8, 48)
        end = rng.randint(3, 40)
    else:  # fabricated / stuffed
        prof = rng.choices(["advanced", "expert"], [2, 8])[0]
        dur = rng.choice([0, 0, 0, 2])
        end = rng.randint(0, 3)
    return {"name": name, "proficiency": prof, "endorsements": end, "duration_months": dur}


def _signals(rng, fake, active=True, salary=(20, 50)):
    if active:
        la = rng.randint(0, 14); rr = round(rng.uniform(0.55, 0.95), 2); otw = True
        npd = rng.choice([0, 15, 30, 45]); ic = round(rng.uniform(0.8, 1.0), 2)
    else:
        la = rng.randint(120, 400); rr = round(rng.uniform(0.0, 0.2), 2); otw = False
        npd = rng.choice([60, 90, 120]); ic = round(rng.uniform(0.3, 0.6), 2)
    today = date(2026, 6, 27)
    return {
        "profile_completeness_score": rng.randint(60, 100),
        "signup_date": _iso(today - timedelta(days=rng.randint(60, 1200))),
        "last_active_date": _iso(today - timedelta(days=la)),
        "open_to_work_flag": otw,
        "profile_views_received_30d": rng.randint(0, 300),
        "applications_submitted_30d": rng.randint(0, 12),
        "recruiter_response_rate": rr,
        "avg_response_time_hours": round(rng.uniform(1, 120), 1),
        "skill_assessment_scores": {},
        "connection_count": rng.randint(50, 5000),
        "endorsements_received": rng.randint(0, 200),
        "notice_period_days": npd,
        "expected_salary_range_inr_lpa": {"min": salary[0], "max": salary[1]},
        "preferred_work_mode": rng.choice(["remote", "hybrid", "onsite", "flexible"]),
        "willing_to_relocate": rng.random() < 0.6,
        "github_activity_score": rng.choice([-1] + list(range(0, 100))),
        "search_appearance_30d": rng.randint(0, 500),
        "saved_by_recruiters_30d": rng.randint(0, 40),
        "interview_completion_rate": ic,
        "offer_acceptance_rate": rng.choice([-1, round(rng.uniform(0.2, 1.0), 2)]),
        "verified_email": rng.random() < 0.9,
        "verified_phone": rng.random() < 0.7,
        "linkedin_connected": rng.random() < 0.8,
    }


def _education(rng):
    return [{"institution": rng.choice(INSTITUTIONS), "degree": "B.Tech",
             "field_of_study": rng.choice(["Computer Science", "IT", "Electronics"]),
             "start_year": 2012, "end_year": 2016,
             "grade": None, "tier": rng.choices(TIERS, [8, 22, 40, 25, 5])[0]}]


def _make(rng, fake, idx, archetype, tier):
    cid = f"CAND_{idx:07d}"
    end = date(2026, 6, 27)
    honeypot = archetype == "honeypot"

    if archetype in ("perfect", "strong"):
        yoe = round(rng.uniform(6, 8.5) if archetype == "perfect" else rng.uniform(5, 9), 1)
        title = rng.choice(AI_TITLES)
        groups = list(GROUP_SKILLS) if archetype == "perfect" else rng.sample(list(GROUP_SKILLS), rng.randint(5, 7))
        skills = [_skill(rng, rng.choice(GROUP_SKILLS[g]), True) for g in groups]
        skills += [_skill(rng, s, True) for s in rng.sample(NICE_SKILLS, rng.randint(1, 3))]
        cos = PRODUCT_COS if archetype == "perfect" else (PRODUCT_COS * 2 + SERVICES_COS)
        industry = "AI/ML"
        desc = PROD_DESC.format(x=rng.choice(["search", "recommendation", "ranking"]))
        active = True
    elif archetype == "plain_tier5":  # rich career text, keyword-light skills
        yoe = round(rng.uniform(6, 8.5), 1)
        title = rng.choice(["Engineering Lead", "Principal Engineer", "Staff Engineer"])
        skills = [_skill(rng, s, True) for s in rng.sample(["Python", "Distributed Systems",
                  "SQL", "Scala", "Java", "Microservices"], 4)]
        cos = PRODUCT_COS
        industry = rng.choice(["Software", "E-commerce", "Fintech"])
        desc = PLAIN_DESC
        active = True
    elif archetype == "good":
        yoe = round(rng.uniform(4, 10), 1)
        title = rng.choice(["Data Scientist", "ML Engineer", "Software Engineer"])
        groups = rng.sample(list(GROUP_SKILLS), rng.randint(2, 4))
        skills = [_skill(rng, rng.choice(GROUP_SKILLS[g]), True) for g in groups]
        skills += [_skill(rng, s, True) for s in rng.sample(OFF["Backend Engineer"], 2)]
        cos = PRODUCT_COS + SERVICES_COS + OTHER_COS
        industry = rng.choice(["Software", "Fintech", "E-commerce"])
        desc = "Built data and ML models; some retrieval and ranking exposure."
        active = rng.random() < 0.7
    elif archetype == "keyword_stuffer":  # non-AI title, AI skills listed
        yoe = round(rng.uniform(3, 12), 1)
        title = rng.choice(["Marketing Manager", "HR Manager", "Sales Executive", "Content Writer"])
        skills = [_skill(rng, rng.choice(GROUP_SKILLS[g]), rng.random() < 0.3)
                  for g in rng.sample(list(GROUP_SKILLS), rng.randint(5, 8))]
        cos = SERVICES_COS + OTHER_COS
        industry = rng.choice(["IT Services", "Manufacturing", "Conglomerate"])
        desc = OFF_DESC.format(x=title.lower())
        active = rng.random() < 0.6
    elif archetype == "honeypot":
        yoe = round(rng.uniform(2, 6), 1)
        title = rng.choice(AI_TITLES)
        # expert in many skills, mostly 0 months used
        groups = list(GROUP_SKILLS)
        skills = [_skill(rng, rng.choice(GROUP_SKILLS[g]), False) for g in groups]
        skills += [_skill(rng, s, False) for s in NICE_SKILLS[:4]]
        cos = OTHER_COS
        industry = "AI/ML"
        desc = "Expert in embeddings, FAISS, RAG, BM25, NDCG, vector search, reranking."
        active = True
    else:  # moderate / weak / irrelevant -> off-target
        yoe = round(rng.uniform(1, 16), 1)
        offtitle = rng.choice(list(OFF))
        title = offtitle
        skills = [_skill(rng, s, True) for s in OFF[offtitle]]
        if archetype == "weak":
            skills += [_skill(rng, rng.choice(GROUP_SKILLS["python"]), True)]
        cos = SERVICES_COS + OTHER_COS
        industry = rng.choice(["IT Services", "Manufacturing", "Paper Products", "Conglomerate"])
        desc = OFF_DESC.format(x=offtitle.lower())
        active = rng.random() < 0.5

    # build 2-4 career rows summing ~ yoe
    n_jobs = max(1, min(4, round(yoe / 2.5)))
    months_each = max(6, int(yoe * 12 / n_jobs))
    career, cur_end = [], end
    for k in range(n_jobs):
        job = _job(rng, title if k == 0 else title, rng.choice(cos), industry,
                   months_each, cur_end, desc, honeypot=(honeypot and k == 0))
        career.append(job)
        cur_end = cur_end - timedelta(days=int(months_each * 30.4) + 30)

    rec = {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": fake.name(),
            "headline": f"{title} | {', '.join(s['name'] for s in skills[:3])}",
            "summary": desc + f" {yoe:.0f} years of experience.",
            "location": rng.choice(LOCS), "country": "India",
            "years_of_experience": yoe, "current_title": title,
            "current_company": career[0]["company"],
            "current_company_size": career[0]["company_size"],
            "current_industry": industry,
        },
        "career_history": career,
        "education": _education(rng),
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": _signals(rng, fake, active=active),
    }
    return rec, tier


def generate(n: int = 5000, out_dir: str | Path = "data/synth", seed: int = 42):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cand_path, gt_path = out / "candidates.jsonl", out / "ground_truth.csv"
    rng = random.Random(seed); fake = Faker(); Faker.seed(seed)
    names = list(ARCHE); weights = [ARCHE[a][0] for a in names]

    with cand_path.open("w") as cf, gt_path.open("w", newline="") as gf:
        gw = csv.writer(gf); gw.writerow(["candidate_id", "relevance", "is_honeypot", "archetype"])
        i = 0
        while i < n:
            arche = rng.choices(names, weights=weights)[0]
            tier = ARCHE[arche][1]
            rec, t = _make(rng, fake, i, arche, tier)
            cf.write(json.dumps(rec) + "\n")
            gw.writerow([rec["candidate_id"], t, int(arche == "honeypot"), arche]); i += 1
            # behavioral twin: emit a near-identical dormant copy at a lower tier
            if arche in ("perfect", "strong") and rng.random() < 0.25 and i < n:
                twin = json.loads(json.dumps(rec))
                twin["candidate_id"] = f"CAND_{i:07d}"
                twin["redrob_signals"] = _signals(rng, fake, active=False)
                cf.write(json.dumps(twin) + "\n")
                gw.writerow([twin["candidate_id"], max(1, t - 2), 0, "behavioral_twin"]); i += 1
    return cand_path, gt_path


def load_ground_truth(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["candidate_id"]] = {"relevance": int(row["relevance"]),
                                        "is_honeypot": bool(int(row["is_honeypot"])),
                                        "archetype": row["archetype"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    c, g = generate(args.n, args.out, args.seed)
    from collections import Counter
    gt = load_ground_truth(g)
    print(f"Generated {len(gt)} candidates -> {c}")
    print("Tier dist:", dict(Counter(v["relevance"] for v in gt.values())))
    print("Archetypes:", dict(Counter(v["archetype"] for v in gt.values())))
    print("Honeypots:", sum(v["is_honeypot"] for v in gt.values()))
