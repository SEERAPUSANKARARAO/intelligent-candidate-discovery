"""Shared vocabulary for the PoC.

A single source of truth for skills, role families, seniority levels and domains,
used by BOTH the synthetic data generator (``data_gen.py``) and the JD parser
(``jd_parser.py``). Keeping them aligned is what makes skill matching meaningful.

Each skill maps to a list of aliases/surface forms so the JD parser can recognise
paraphrases ("ML" == "machine learning") — this is the cheap-but-effective backbone
that the embedding layer then complements for true semantic matching.
"""

from __future__ import annotations

# canonical skill -> surface forms (lowercase). The canonical form is the dict key.
SKILL_ALIASES: dict[str, list[str]] = {
    # --- backend / general SE ---
    "python": ["python", "py"],
    "java": ["java"],
    "go": ["golang", "go lang", " go "],
    "node.js": ["node", "nodejs", "node.js"],
    "typescript": ["typescript", "ts"],
    "rust": ["rust"],
    "c++": ["c++", "cpp"],
    "sql": ["sql"],
    "postgresql": ["postgres", "postgresql", "psql"],
    "mongodb": ["mongo", "mongodb"],
    "redis": ["redis"],
    "graphql": ["graphql"],
    "rest apis": ["rest", "restful", "rest api", "rest apis"],
    "microservices": ["microservice", "microservices", "micro-services"],
    "distributed systems": ["distributed systems", "distributed system"],
    "kafka": ["kafka"],
    "rabbitmq": ["rabbitmq"],
    # --- cloud / devops ---
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    "docker": ["docker", "containerization", "containerisation"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous delivery"],
    "linux": ["linux"],
    # --- data / ml ---
    "machine learning": ["machine learning", "ml", "ml models"],
    "deep learning": ["deep learning", "dl", "neural networks", "neural network"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv", "image recognition"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pandas": ["pandas"],
    "spark": ["spark", "pyspark", "apache spark"],
    "airflow": ["airflow"],
    "data engineering": ["data engineering", "data pipelines", "etl"],
    "data analysis": ["data analysis", "data analytics", "analytics"],
    "llm": ["llm", "large language models", "generative ai", "gen ai", "genai"],
    # --- frontend ---
    "react": ["react", "react.js", "reactjs"],
    "vue": ["vue", "vue.js", "vuejs"],
    "angular": ["angular"],
    "css": ["css", "scss", "tailwind"],
    "html": ["html"],
    "ui/ux": ["ui/ux", "ux", "user experience", "ui design"],
    # --- product / soft / misc ---
    "agile": ["agile", "scrum", "kanban"],
    "product management": ["product management", "product manager", "roadmap"],
    "leadership": ["leadership", "team lead", "mentoring", "people management"],
    "communication": ["communication", "stakeholder management"],
    "system design": ["system design", "architecture", "software architecture"],
}

ALL_SKILLS: list[str] = list(SKILL_ALIASES.keys())

# role families bundle a coherent set of skills; data_gen draws a candidate's
# skills mostly from one family (+ a little noise) so profiles are realistic.
ROLE_FAMILIES: dict[str, dict] = {
    "Backend Engineer": {
        "core": ["python", "java", "go", "sql", "postgresql", "rest apis",
                 "microservices", "distributed systems", "redis", "kafka", "system design"],
        "titles": ["Backend Engineer", "Software Engineer", "Backend Developer",
                   "Senior Software Engineer", "Staff Engineer"],
    },
    "Frontend Engineer": {
        "core": ["typescript", "react", "vue", "angular", "css", "html",
                 "ui/ux", "node.js", "graphql"],
        "titles": ["Frontend Engineer", "UI Engineer", "Frontend Developer",
                   "Web Developer", "Senior Frontend Engineer"],
    },
    "ML Engineer": {
        "core": ["python", "machine learning", "deep learning", "pytorch",
                 "tensorflow", "nlp", "computer vision", "scikit-learn", "llm",
                 "spark", "data engineering"],
        "titles": ["ML Engineer", "Machine Learning Engineer", "AI Engineer",
                   "Research Engineer", "Applied Scientist"],
    },
    "Data Engineer": {
        "core": ["python", "sql", "spark", "airflow", "data engineering",
                 "aws", "kafka", "data analysis", "postgresql"],
        "titles": ["Data Engineer", "Analytics Engineer", "Big Data Engineer",
                   "Senior Data Engineer"],
    },
    "Data Scientist": {
        "core": ["python", "machine learning", "data analysis", "pandas",
                 "scikit-learn", "nlp", "sql", "deep learning"],
        "titles": ["Data Scientist", "Senior Data Scientist", "Applied Scientist"],
    },
    "DevOps Engineer": {
        "core": ["aws", "gcp", "azure", "docker", "kubernetes", "terraform",
                 "ci/cd", "linux", "python"],
        "titles": ["DevOps Engineer", "SRE", "Platform Engineer",
                   "Infrastructure Engineer", "Cloud Engineer"],
    },
    "Product Manager": {
        "core": ["product management", "agile", "communication", "leadership",
                 "data analysis", "ui/ux"],
        "titles": ["Product Manager", "Senior Product Manager", "Group Product Manager"],
    },
}

# seniority ladder with the typical years-of-experience band each implies.
SENIORITY_LEVELS: list[str] = ["intern", "junior", "mid", "senior", "lead", "principal"]
SENIORITY_RANK: dict[str, int] = {lvl: i for i, lvl in enumerate(SENIORITY_LEVELS)}
SENIORITY_YEARS: dict[str, tuple[int, int]] = {
    "intern": (0, 1),
    "junior": (1, 3),
    "mid": (3, 6),
    "senior": (6, 10),
    "lead": (9, 14),
    "principal": (12, 25),
}
SENIORITY_ALIASES: dict[str, list[str]] = {
    "intern": ["intern", "internship"],
    "junior": ["junior", "jr", "entry level", "entry-level", "associate"],
    "mid": ["mid", "mid-level", "intermediate"],
    "senior": ["senior", "sr"],
    "lead": ["lead", "team lead", "tech lead", "leads"],
    "principal": ["principal", "staff", "distinguished", "architect"],
}

DOMAINS: dict[str, list[str]] = {
    "fintech": ["fintech", "financial", "finance", "banking", "payments", "trading"],
    "healthcare": ["healthcare", "health", "medical", "biotech", "pharma"],
    "e-commerce": ["e-commerce", "ecommerce", "retail", "marketplace"],
    "social media": ["social media", "social network", "social"],
    "gaming": ["gaming", "games", "game studio"],
    "saas": ["saas", "b2b", "enterprise software"],
    "edtech": ["edtech", "education", "e-learning"],
    "logistics": ["logistics", "supply chain", "delivery"],
}
ALL_DOMAINS: list[str] = list(DOMAINS.keys())

EDUCATION_LEVELS: list[str] = ["Bootcamp", "Bachelor's", "Master's", "PhD"]
CERTIFICATIONS: list[str] = [
    "AWS Certified Solutions Architect", "Google Cloud Professional",
    "Certified Kubernetes Administrator", "PMP", "Certified Scrum Master",
    "TensorFlow Developer Certificate", "Azure Solutions Architect",
]


def _contains_alias(text_lower: str, aliases: list[str]) -> bool:
    """True if any alias appears as a token-ish substring in ``text_lower``."""
    for alias in aliases:
        a = alias.strip()
        if not a:
            continue
        # pad to reduce false positives on very short aliases (e.g. "go", "ts")
        if len(a) <= 3:
            if f" {a} " in f" {text_lower} " or text_lower == a:
                return True
        elif a in text_lower:
            return True
    return False


def find_skills(text: str) -> list[str]:
    """Return canonical skills whose aliases appear in ``text``."""
    tl = text.lower()
    return [skill for skill, aliases in SKILL_ALIASES.items() if _contains_alias(tl, aliases)]


def find_domains(text: str) -> list[str]:
    tl = text.lower()
    return [dom for dom, aliases in DOMAINS.items() if _contains_alias(tl, aliases)]
