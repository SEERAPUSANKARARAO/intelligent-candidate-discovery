# Intelligent Candidate Discovery — Redrob AI · India Runs (Track 1)

> Rank the **top-100** candidates for *Senior AI Engineer (Founding Team)* from the
> **100,000-row** `candidates.jsonl` — **offline, CPU-only, ~80 s, <2.5 GB, zero
> honeypots**, with an evidence-cited reason for every pick. Passes the official
> validator; built against the real schema, JD and spec.

This is not a keyword filter and not a black box. It is a transparent retrieval
funnel that reasons about the **gap between what a profile says and what it means**
(the JD's words), rejects the dataset's four traps, integrates the 23 behavioural
signals, and *proves* its quality with a labelled harness.

---

## 1. The problem behind the problem
The data is adversarial: ~30% are IT-services candidates, real AI titles are rare
(~278 "AI/ML" profiles in 100k), and there are **four traps** — keyword-stuffers
(AI skills on a non-AI career), plain-language Tier-5s (real builders who never
write "RAG"), behavioral-twins (same profile, different engagement), and ~80
honeypots (impossible dates / unused "expert" skills). Naive approaches fail:

| Approach | Composite | Honeypots in top-100 |
|---|---:|---:|
| BM25 keyword-only | 0.58 | 100% → ❌ DQ |
| Embedding-only | 0.10 | 87% → ❌ DQ |
| **This system (full)** | **0.93** | **0% → ✅** |

Composite = `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` (official).

---

## 2. Architecture — 4-stage offline funnel
```
 100k candidates (JSONL, streamed, schema-adapted)
   │  STAGE 1 · trust-aware STRUCTURED pre-score (ALL candidates)
   │    skills×trust(endorse/duration/proficiency) · career(title×product-vs-services×narrative) · experience
   │    → keyword density is NOT used → honeypots & stuffers can't crowd recall
   ▼  STAGE 2 · recall: top ~2,000
   ▼  STAGE 3 · hybrid rerank (shortlist): BM25 ⊕ semantic (TF-IDF default / precomputed BGE)
   ▼  STAGE 4 · fuse → behavioural multiplier (23 signals) → JD disqualifier penalties
   │    → plain-language Tier-5 rescue → honeypot floor
   ▼  submission.csv  (candidate_id,rank,score,reasoning; tie-break candidate_id asc)
```

**Key ideas**
- **Trust over keywords** — a self-claimed "expert, 0 months, 0 endorsements" skill
  collapses to ~0.2 trust vs >0.7 corroborated; stuffers gain nothing.
- **Reason about JD intent** — career evidence credits a skill group even with no
  keyword (rescues Tier-5 product builders); a Marketing Manager with every AI skill
  still loses (no title/career corroboration).
- **Disqualifiers** the JD names — services-only, research-only, framework-only,
  job-hopping — apply explicit penalties.
- **Signal integration** — 23 behavioural signals → ×[0.50,1.20] multiplier; dormant,
  unresponsive candidates are correctly down-weighted.
- **Honeypot floor** — date/duration impossibility + unused expert-skill breadth →
  floored out of the top-100 (the hard DQ gate), staying conservative to avoid
  dropping real candidates.

Text matching uses **token-set membership** (word boundaries for free), so `"ai"`
never matches "ret**ai**l" and `"rag"` never matches "sto**rag**e".

---

## 3. Reproduce
```bash
pip install -r requirements.txt
# (optional) regenerate the labelled harness:
python -m challenge.synth --n 5000 --out data/synth

# produce the submission (DEFAULT: TF-IDF, no heavy deps, ~80s on 100k):
python rank.py --candidates data/candidates.jsonl --out submission.csv
python data/official_validate_submission.py submission.csv          # "Submission is valid."

# measure locally (ablation + official composite + CIs + per-trap):
python -m challenge.evaluate --in data/synth/candidates.jsonl --gt data/synth/ground_truth.csv

# rigour:
python -m challenge.adversarial            # honeypot red/blue catch-rates
python -m challenge.fairness --candidates data/demo_candidates.jsonl

# interactive product UI / sandbox:
uvicorn backend.main:app --port 8000       # http://localhost:8000
```

### Optional depth (opt-in; pre-computed offline, allowed by the rules)
```bash
# 1) precompute BGE embeddings + FAISS (once, offline) → artifacts the ranker loads fast
python -m challenge.embed_index --candidates data/candidates.jsonl --out challenge/artifacts
# 2) train LambdaMART on the labelled harness
python -m challenge.ltr --train data/synth/candidates.jsonl --gt data/synth/ground_truth.csv
# 3) rank with any combination (still offline / CPU / in-budget)
python rank.py --candidates data/candidates.jsonl --out submission.csv \
    --embeddings challenge/artifacts --ltr --rerank
```
The default ship is the **interpretable fusion** (no overfit to synthetic labels);
BGE/LambdaMART/cross-encoder are demonstrated, measured refinements.

### Docker (Stage-3 reproduction, offline)
```bash
docker build -t redrob-ranker .
docker run --rm --network none -m 16g -v "$PWD/data:/data" redrob-ranker \
    --candidates /data/candidates.jsonl --out /data/submission.csv
```

---

## 4. Files
```
challenge/
  schema.py        real→canonical adapter + safe/gzip loader
  jd.py            real JD signals (must/nice, disqualifiers, industry-based services)
  features.py      one feature extractor (token-set matching, product-over-keywords)
  scoring.py       fusion + disqualifier penalties + honeypot floor
  honeypot.py      date/duration consistency detection (calibrated on real data)
  retrieval.py     BM25 (scipy) + TF-IDF / BGE semantic + hybrid blend
  reasoning.py     Stage-4 reasoning (specific · varied · honest · no hallucination)
  metrics.py       NDCG@10/@50 · MAP · P@10 · official composite
  rank.py          the 4-stage funnel → submission.csv   (root rank.py delegates here)
  evaluate.py      ablation + official composite + bootstrap CIs + per-trap breakdown
  synth.py         labelled harness (real schema, tiers 0-5, four traps)
  embed_index.py   offline BGE + FAISS precompute (opt-in)
  rerank.py        cross-encoder rerank (opt-in)
  ltr.py           LambdaMART learning-to-rank (opt-in)
  fairness.py      bias audit + blind mode
  adversarial.py   honeypot red/blue evasion loop
  validate_submission.py   official-format checker
backend/ + frontend/   FastAPI + SPA product UI (doubles as the required sandbox)
tests/test_challenge.py  metrics · trust · honeypot · token-matching · stuffer · e2e
```

**Constraints honoured:** offline · CPU-only · ~80 s / <2.5 GB on 100k · deterministic ·
exact submission format + official tie-break. Default needs only
`numpy`/`scipy`/`scikit-learn`.
