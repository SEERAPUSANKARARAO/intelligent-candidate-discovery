"""The real target role, as a machine-readable signal spec.

Encodes the *actual* Redrob JD (Senior AI Engineer — Founding Team), including the
parts most entries ignore: the explicit disqualifiers and "do NOT want" list, and
the instruction to reason about the gap between what the JD says and what it means
(a product-company recsys builder with no AI keywords beats a "Marketing Manager"
with a perfect keyword list). Everything here is data, shared by scoring,
reasoning, the synthetic generator and the demo.
"""

from __future__ import annotations

JD_TITLE = "Senior AI Engineer (Founding Team)"
JD_COMPANY = "Redrob AI"

JD_TEXT = (
    "Redrob AI (Series A) is building a new AI Engineering org and needs a founding "
    "Senior AI Engineer to own the intelligence layer: retrieval, ranking and "
    "matching. You must have PRODUCTION experience with embeddings-based retrieval "
    "(sentence-transformers, BGE, E5, OpenAI), vector databases / hybrid search "
    "(FAISS, Qdrant, Milvus, Pinecone, Elasticsearch, OpenSearch), strong Python, "
    "and rigorous ranking evaluation (NDCG, MRR, MAP, A/B testing, offline-online "
    "correlation). 5-9 years, ideally 6-8 with 4-5 in applied ML at PRODUCT "
    "companies (not pure services), having shipped an end-to-end ranking/search/"
    "recommendation system at real scale. Tilt toward shipper over researcher. "
    "Nice to have: LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR-tech, "
    "distributed systems. Based in / willing to relocate to Noida or Pune."
)

# canonical group -> surface forms (lowercase). A candidate "covers" a group if any
# form appears in their skills OR is corroborated by career text.
MUST_HAVE: dict[str, list[str]] = {
    "embeddings": ["embedding", "embeddings", "sentence-transformers", "sentence transformers",
                   "bge", " e5", "openai embeddings", "dense retrieval", "dense vector",
                   "text embedding", "gte", "word2vec", "fasttext"],
    "vector_db": ["faiss", "pinecone", "milvus", "qdrant", "weaviate", "opensearch",
                  "elasticsearch", "chroma", "pgvector", "vespa", "vector database",
                  "vector store", "ann", "hnsw", "vector search"],
    "ranking_ir": ["bm25", "tf-idf", "tfidf", "okapi", "hybrid search", "hybrid retrieval",
                   "information retrieval", "reranking", "re-ranking", "reranker",
                   "cross-encoder", "learning to rank", "ltr", "ranking", "relevance"],
    "evaluation": ["ndcg", "mrr", "mean average precision", " map", "recall@", "precision@",
                   "a/b test", "ab test", "offline evaluation", "relevance evaluation",
                   "experimentation"],
    "python": ["python", "pytorch", "scikit-learn", "numpy", "pandas"],
    "nlp_llm": ["nlp", "natural language processing", "transformer", "transformers", "bert",
                "gpt", "llm", "large language model", "rag", "retrieval augmented",
                "text classification", "named entity", "semantic search", "question answering"],
    "ml_core": ["machine learning", "deep learning", "neural network", "tensorflow",
                "ml model", "model training", "feature engineering", "classifier"],
    "search_rec": ["search", "recommendation", "recommender", "recsys", "collaborative filtering",
                   "candidate generation", "two-tower", "matching", "personalization",
                   "solr", "lucene"],
}

NICE_TO_HAVE: dict[str, list[str]] = {
    "llm_finetuning": ["fine-tuning", "finetuning", "lora", "qlora", "peft", "rlhf", "dpo",
                       "instruction tuning"],
    "learning_to_rank": ["learning to rank", "lambdamart", "xgboost", "lightgbm", "ranknet",
                         "listwise", "ranker"],
    "mlops": ["mlops", "mlflow", "kubeflow", "model serving", "triton", "bentoml", "docker",
              "kubernetes", "ci/cd", "model registry", "inference optimization"],
    "data_eng": ["spark", "airflow", "etl", "data pipeline", "kafka", "dbt", "flink", "warehouse"],
    "hrtech": ["hr-tech", "hrtech", "recruiting", "marketplace", "talent", "ats"],
}

# --- career trajectory ----------------------------------------------------------
IDEAL_TITLES: list[str] = [
    "ai engineer", "machine learning engineer", "ml engineer", "senior ml engineer",
    "senior machine learning engineer", "nlp engineer", "search engineer",
    "information retrieval engineer", "ir engineer", "applied scientist",
    "research engineer", "recommendation systems engineer", "search relevance engineer",
    "staff machine learning engineer", "applied ml engineer", "data scientist",
    "ranking engineer", "relevance engineer",
]
TITLE_KEYWORDS: list[str] = [
    "ai", "ml", "machine learning", "nlp", "search", "retrieval", "relevance",
    "recommendation", "applied scientist", "research engineer", "data scientist",
]
# JD-relevant phrases that show real retrieval/ranking work in role descriptions
CAREER_KEYWORDS: list[str] = [
    "embedding", "vector", "faiss", "retrieval", "ranking", "rerank", "bm25", "semantic search",
    "search relevance", "ndcg", "recall", "rag", "llm", "transformer", "bert", "recommendation",
    "recommender", "ann", "hybrid search", "fine-tune", "pytorch", "information retrieval",
    "learning to rank", "personalization", "matching", "relevance", "query understanding",
]
# strong evidence the work actually shipped (anti-honeypot positive signal)
PRODUCTION_EVIDENCE: list[str] = [
    "production", "deployed", "shipped", "scaled", "latency", "throughput", "served",
    "millions", "billions", "a/b test", "real-time", "live traffic", "users", "scale",
    "improved", "increased", "reduced", "optimized", "launched",
]

# --- company / industry classification ------------------------------------------
# Services/consulting (JD: "do NOT want people who only worked at consulting firms")
SERVICES_INDUSTRIES: set[str] = {"it services", "consulting", "outsourcing", "staffing"}
SERVICES_COMPANIES: set[str] = {
    "tcs", "tata consultancy", "infosys", "wipro", "cognizant", "accenture", "capgemini",
    "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis", "hexaware", "birlasoft",
    "persistent", "coforge", "deloitte", "ibm", "dxc", "genpact", "l&t infotech",
}
# product / software industries that count toward "product company" experience
PRODUCT_INDUSTRIES: set[str] = {
    "software", "fintech", "food delivery", "e-commerce", "ecommerce", "saas", "ai/ml",
    "edtech", "healthtech", "social media", "gaming", "logistics", "mobility",
}
PRODUCT_COMPANIES: set[str] = {
    "google", "microsoft", "amazon", "meta", "netflix", "apple", "nvidia", "openai",
    "anthropic", "flipkart", "zomato", "swiggy", "cred", "razorpay", "sarvam ai", "sarvam",
    "paytm", "phonepe", "sprinklr", "haptik", "freshworks", "postman", "zoho", "dunzo",
    "meesho", "groww", "uber", "linkedin", "salesforce", "adobe", "atlassian", "databricks",
    "cohere", "perplexity", "glance", "inmobi", "ola", "myntra", "navi", "slice", "sarvam",
}

# things the JD explicitly does NOT want
RESEARCH_ONLY_HINTS = ["phd researcher", "research scientist", "postdoc", "academic",
                       "university", "lab", "publication", "thesis"]
WRONG_DOMAIN_HINTS = ["computer vision", "image", "speech", "robotics", "autonomous driving",
                      "lidar", "perception", "slam"]
FRAMEWORK_ONLY_HINTS = ["langchain", "llama-index", "llamaindex", "autogpt", "chatbot demo"]

# experience band
YOE_IDEAL = (6.0, 8.0)
YOE_ACCEPTABLE = (5.0, 9.0)

# location preference (India tier-1; relocation considered)
PREFERRED_LOCATIONS = ["noida", "pune", "delhi", "gurgaon", "gurugram", "hyderabad",
                       "mumbai", "bangalore", "bengaluru", "chennai", "ncr"]

TOP_K = 100


def must_have_groups() -> list[str]:
    return list(MUST_HAVE.keys())
