---
title: Intelligent Candidate Discovery
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Redrob · Intelligent Candidate Discovery (India Runs — Track 1)

> The header block above is HuggingFace Space config (used only when deployed as a
> Docker Space); it is harmless on GitHub.

Rank the **top-100** candidates for *Senior AI Engineer (Founding Team)* at Redrob AI
from a **100,000-row** `candidates.jsonl` — **offline, CPU-only, ~80 s, <2.5 GB,
0 honeypots** — with an evidence-cited reason for every pick. Passes the official
validator; built against the real schema, JD, and submission spec.

---

## What's here

| Part | Path | What it is |
|---|---|---|
| **Ranking engine (the submission)** | [`challenge/`](challenge/README.md) | The scored deliverable: `candidates.jsonl → submission.csv`. 4-stage offline funnel (trust-aware structured recall → hybrid BM25⊕semantic rerank → behavioural fusion → honeypot floor), labelled eval harness, optional depth (BGE/FAISS/LambdaMART/cross-encoder). |
| **Product UI / sandbox** | `backend/` + `frontend/` | FastAPI API + a recruiter-cockpit SPA on the same engine — the required hosted sandbox (run on a ≤100 sample). |

👉 **Judges: start with [`challenge/README.md`](challenge/README.md)** (architecture +
rationale), [`METHODOLOGY.md`](METHODOLOGY.md) (≤200-word summary), and
[`challenge/RESULTS.md`](challenge/RESULTS.md) (ablation + confidence intervals).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Produce the submission (DEFAULT: TF-IDF + BM25 + structured — no heavy deps)
python rank.py --candidates data/candidates.jsonl --out submission.csv

# 2) Validate it against the official rules
python data/official_validate_submission.py submission.csv      # "Submission is valid."

# 3) Measure locally (ablation + official composite + CIs + per-trap)
python -m challenge.synth   --n 20000 --out data/synth20k
python -m challenge.evaluate --in data/synth20k/candidates.jsonl --gt data/synth20k/ground_truth.csv

# 4) Run the product UI / sandbox
uvicorn backend.main:app --port 8000            # → http://localhost:8000
```

The ranking step is fully offline (no network, no GPU) and finishes in ~80 s on the
real 100k pool within 2.5 GB.

### Optional depth (opt-in; pre-computed offline, which the rules allow)
```bash
python -m challenge.embed_index --candidates data/candidates.jsonl --out challenge/artifacts  # BGE + FAISS
python -m challenge.ltr --train data/synth/candidates.jsonl --gt data/synth/ground_truth.csv  # LambdaMART
python rank.py --candidates data/candidates.jsonl --out submission.csv --embeddings challenge/artifacts --ltr --rerank
```
The shipped default is the interpretable fusion (no overfit); these are measured refinements.

### Docker
```bash
# Offline ranking (Stage-3 reproduction):
docker build -f Dockerfile.rank -t redrob-ranker .
docker run --rm --network none -m 16g -v "$PWD/data:/data" redrob-ranker \
    --candidates /data/candidates.jsonl --out /data/submission.csv

# Product UI / sandbox (also the HuggingFace Space image):
docker build -t redrob-ui . && docker run --rm -p 7860:7860 redrob-ui   # → http://localhost:7860
```

### Deploy the sandbox on HuggingFace Spaces
Create a **Docker** Space, then push this repo to it (the `Dockerfile` + the README
front-matter make it serve the UI on port 7860 automatically):
```bash
pip install -U huggingface_hub && huggingface-cli login
huggingface-cli repo create intelligent-candidate-discovery --type space --space_sdk docker
git remote add space https://huggingface.co/spaces/<HF_USER>/intelligent-candidate-discovery
git push space main
```

---

## Product UI / API

`uvicorn backend.main:app` serves the SPA at `/` and this API (loads a small demo pool;
set `DEMO_DATA` to override):

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/jd` | the target role + skill groups |
| `GET`  | `/api/stats` | pool size, honeypots detected, strong count, engine |
| `POST` | `/api/rank` | `{weights, use_semantic, use_behavioral, use_honeypot, top_n, search}` → ranked cards |
| `GET`  | `/api/candidate/{id}` | full profile + score breakdown + reasoning + honeypot verdict |
| `GET`  | `/api/traps` | honeypots + keyword-stuffers detected and excluded |

The UI shows live signal-weight sliders, pipeline toggles, score gauges + sub-score bars,
evidence-cited reasoning, a candidate drawer (skills with trust, career timeline, 23
signals), and a "traps caught" view. A Streamlit sandbox is also available:
`streamlit run challenge/app.py`.

---

## Results (20k labelled synthetic harness)

| Configuration | Composite | NDCG@10 | NDCG@50 | Honeypots@100 | DQ? |
|---|---:|---:|---:|---:|:--:|
| BM25 keyword-only | 0.587 | 0.636 | 0.555 | 31% | ❌ |
| Embedding-only | 0.004 | 0.000 | 0.000 | 79% | ❌ |
| Structured (trust) | 0.810 | 0.781 | 0.764 | 0% | ✅ |
| + semantic (hybrid) | 0.895 | 0.895 | 0.840 | 0% | ✅ |
| + behavioural | 0.920 | 0.921 | 0.884 | 2% | ✅ |
| **FULL** | **0.933** | **0.921** | **0.908** | **0%** | ✅ |

Composite = `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` (official). FULL: 95% CI
0.873–0.956; **0/600 keyword-stuffers and 0/372 honeypots** in the top-100. See
[`challenge/RESULTS.md`](challenge/RESULTS.md). (Synthetic proxy — real ground truth is hidden.)

---

## Tests
```bash
pytest -q          # 16 tests: metrics · trust · honeypot · token-matching · stuffer · end-to-end
```

## Repo map
```
rank.py                  root entrypoint (official reproduce command)
challenge/               the ranking engine + eval/synth/depth/fairness/adversarial (see challenge/README.md)
backend/                 FastAPI app (main.py) + engine wrapper (service.py)
frontend/                product SPA (index.html, styles.css, app.js)
data/                    candidates.jsonl (symlink) + official validator + samples
Dockerfile, requirements*.txt, submission_metadata.yaml, METHODOLOGY.md
```
