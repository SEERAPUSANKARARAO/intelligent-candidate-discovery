# Evaluation results — Redrob India Runs (Track 1)

Reproducible measurement of the ranker against the **official composite**
(`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`) and the honeypot DQ gate,
on a **20,000-candidate labelled synthetic harness** (real schema, relevance tiers
0–5, all four documented traps).

```bash
python -m challenge.synth   --n 20000 --out data/synth20k --seed 42
python -m challenge.evaluate --in data/synth20k/candidates.jsonl --gt data/synth20k/ground_truth.csv
```

> **Why synthetic?** The real ground truth is hidden and only 3 submissions are
> allowed, so we built a labelled proxy to measure the metric locally. Trust the
> **relative lift over baselines**, the **0% honeypot rate**, and the **trap
> rejection** — not the absolute number as a leaderboard guarantee.

## Harness composition (20k)
Tiers: `0:12118  1:4018  2:2131  3:1008  4:503  5:222`
Archetypes: irrelevant 11146 · weak 4018 · moderate 1993 · good 984 · keyword_stuffer 600 ·
strong 503 · honeypot 372 · behavioral_twin 162 · plain_tier5 118 · perfect 104.

## Ablation — each row adds one capability

| Configuration | Composite | NDCG@10 | NDCG@50 | MAP | P@10 | Honeypots@100 | DQ? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 1. BM25 keyword-only | 0.5874 | 0.6358 | 0.5545 | 0.4546 | 0.70 | 31% | ❌ |
| 2. Embedding-only | 0.0040 | 0.0000 | 0.0000 | 0.0266 | 0.00 | 79% | ❌ |
| 3. Structured (trust skill/career/exp) | 0.8099 | 0.7814 | 0.7641 | 1.0000 | 0.80 | **0%** | ✅ |
| 4. + semantic (hybrid) | 0.8945 | 0.8948 | 0.8402 | 1.0000 | 0.90 | **0%** | ✅ |
| 5. + behavioural signals | 0.9200 | 0.9212 | 0.8839 | 0.9613 | 1.00 | 2% | ✅ |
| **6. + honeypot defence (FULL)** | **0.9329** | **0.9212** | **0.9075** | **1.0000** | **1.00** | **0%** | ✅ |

**FULL system: composite 0.9329, 95% CI 0.873–0.956** (bootstrap, 200 resamples).
Ranked 20k in 6.5 s.

Reading it: keyword and embedding baselines are **disqualified** (the trap works as
designed). Trust scoring rescues both the score *and* the honeypot gate. Behavioural
fusion lifts the top, but can re-admit a couple of honeypots (row 5, 2%); the
honeypot floor (row 6) removes them while keeping the gain.

## Trap rejection (FULL system, per archetype: in top-100 / total)

| Archetype | In top-100 / total | Expected |
|---|---:|---|
| perfect (tier 5) | 38 / 104 | high (capped by 100 slots) |
| strong (tier 4) | 62 / 503 | high |
| **keyword_stuffer** | **0 / 600** | ✅ rejected |
| **honeypot** | **0 / 372** | ✅ rejected (DQ gate: 0%) |
| good / moderate / weak / irrelevant | 0 | ✅ below cutoff |
| plain_tier5 | 0 / 118 | ⚠ hardest case (see caveats) |

The top-100 is saturated by the genuinely best candidates (perfect + strong); **every
keyword-stuffer and every honeypot is excluded.**

## Operational metrics (real 100k pool)
Ground truth is hidden, so no composite — but on the real `candidates.jsonl`:
**ranked in ~62 s, ~2.0 GB RAM, CPU-only, offline, 0 honeypots in top-100, official
`validate_submission.py` passes.**

## Optional depth
- **LambdaMART ensemble** (`--ltr`): composite ≈ 0.996 on synthetic — *overfit to the
  synthetic tiers*, hence opt-in; the shipped default is the interpretable fusion.
- **BGE embeddings** (`--embeddings`) / **cross-encoder** (`--rerank`): offline,
  measured refinements to the semantic stage.

## Caveats
- Numbers are on a **synthetic proxy**, not the hidden real ground truth.
- **MAP = 1.0 / P@10 = 1.0** are synthetic-optimistic (top tiers are cleanly
  separable); **NDCG@50 (0.91)** is the more discriminating signal.
- **Plain-language Tier-5s** (great product builders with keyword-light profiles)
  are the hardest case; a conservative JD-aligned rescue is applied without
  overfitting to the synthetic construction.
