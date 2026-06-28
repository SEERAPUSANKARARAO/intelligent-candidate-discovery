# Methodology — Intelligent Candidate Discovery (Redrob India Runs, Track 1)

## The problem behind the problem
The dataset is adversarial by design. ~30% of candidates sit at IT-services firms,
AI titles are rare (only ~278 "AI/ML" industry profiles in 100k), and there are
**four traps**: keyword-stuffers (AI skills on a non-AI career), plain-language
Tier-5s (real builders who never write "RAG"), behavioral-twins (identical profiles,
different engagement), and ~80 honeypots (impossible dates / unused "expert" skills).
The JD is explicit: *don't reward keyword density — reward the gap between what a
profile says and what it means*, and down-weight unreachable candidates.

## Design: a transparent 4-stage offline funnel
1. **Structured pre-score (all 100k).** A trust-aware score over skills (weighted by
   endorsement/duration/proficiency *consistency*), career trajectory (title fit ×
   product-vs-services × retrieval/ranking narrative), and experience (5–9y band,
   applied-ML fraction). **Keyword density is deliberately not a recall signal**, so
   keyword-stuffers and honeypots can't crowd the shortlist (the failure mode that
   put 96/100 honeypots in the top-100 in an earlier keyword-recall prototype).
2. **Recall.** Keep the top ~2,000 by structured score.
3. **Hybrid rerank (shortlist only).** BM25 ⊕ semantic (TF-IDF default; optional
   precomputed **BGE** embeddings — offline dot-product, no model at rank time).
4. **Fusion + guards.** Weighted fuse → behavioral multiplier from the 23 signals →
   JD disqualifier penalties (services-only, research-only, framework-only,
   job-hopping) → **plain-language Tier-5 rescue** (product-company builders with
   keyword-light profiles) → **honeypot floor** (date/duration impossibility,
   unused expert-skill breadth). Top-100 with evidence-cited reasoning.

## Why it beats keyword/embedding baselines (measured, synthetic harness)
| Approach | Composite | Honeypots in top-100 |
|---|---:|---:|
| BM25 keyword-only | 0.58 | 100% → DQ |
| Embedding-only | 0.10 | 87% → DQ |
| Structured (trust) | 0.87 | 0% |
| + behavioral | 0.92 | 0% |
| **Full** | **0.93** | **0%** |

Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10 (official). Reported
with bootstrap CIs and a per-trap breakdown (all keyword-stuffers & honeypots
rejected; perfect/strong captured).

## Rigour
- **Labelled synthetic harness** (real schema, tiers 0–5, all four traps) — the only
  way to optimise the hidden metric locally given a 3-submission cap.
- **Adversarial honeypot loop** — red-team crafts evasions; we keep the detector
  conservative (false positives drop real candidates and hurt NDCG more than a rare
  borderline miss) while catching the canonical fraud patterns.
- **Fairness audit** — top-100 vs pool selection ratios for institution tier /
  location, plus a "blind" mode; ranking is driven by merit, not pedigree.
- **Stage-4 reasoning** — every claim is verified present in the candidate record
  (automated no-hallucination check), varied, rank-consistent.

## Engineering
Offline, CPU-only, deterministic. ~80s / <2.5 GB on the full 100k (token-set feature
matching avoids the `"ai"`-in-`"retail"` substring trap and keeps recall ~57s).
Exact submission format with the official tie-break; passes the official validator.
Optional **LambdaMART** (NDCG-optimised) and **cross-encoder** reranking are provided
as opt-in refinements; the shipped default is the interpretable fusion (no overfit to
synthetic labels).
