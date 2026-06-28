"""The ranking engine — hybrid retrieval + signal-weighted rerank + explanations.

Two stages:

  Stage 1 (recall, fast): blend semantic cosine and BM25 lexical scores over the
  whole candidate pool to pull a top-K shortlist.

  Stage 2 (precision, explainable): for each shortlisted candidate compute the
  five weighted sub-scores from ``signals.py``, add the recruiter-feedback boost,
  and build a human-readable rationale.

The ``CandidateIndex`` precomputes embeddings + BM25 once at startup so each query
is a couple of matrix ops — "lightning fast" at PoC scale.
"""

from __future__ import annotations

import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from . import embeddings, signals
from .feedback import store as feedback_store
from .jd_parser import parse_jd
from .schemas import (
    Candidate,
    JobRequirements,
    RankedResult,
    RankResponse,
    ScoreComponent,
    SignalWeights,
)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# how many candidates stage 1 forwards to the (more expensive) stage 2 rerank
RETRIEVE_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _candidate_document(c: Candidate) -> str:
    """Text representation of a candidate for embedding + lexical indexing."""
    return (f"{c.current_title}. {c.seniority} {c.role_family} in {c.primary_domain}. "
            f"Skills: {', '.join(c.skills)}. {c.resume_summary}")


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class CandidateIndex:
    """Holds the precomputed retrieval structures for the candidate pool."""

    def __init__(self, candidates: list[Candidate]):
        self.candidates = candidates
        self.by_id = {c.id: c for c in candidates}
        docs = [_candidate_document(c) for c in candidates]
        print(f"[ranking] embedding {len(docs)} candidates...")
        self.embeddings = embeddings.encode(docs)            # (N, dim), normalised
        self.bm25 = BM25Okapi([_tokenize(d) for d in docs])
        print("[ranking] index ready"
              + (" (using fallback embeddings)" if embeddings.is_fallback() else ""))

    # ---- stage 1: hybrid retrieval -------------------------------------------------
    def retrieve(self, req: JobRequirements, k: int = RETRIEVE_K) -> list[tuple[int, float]]:
        """Return [(candidate_index, semantic_score_0_1)] for the top-k blended hits."""
        query_text = req.raw_text + " " + " ".join(req.required_skills + req.nice_to_have_skills)
        q_emb = embeddings.encode([query_text])[0]
        sem = embeddings.cosine_matrix(q_emb, self.embeddings)        # (N,)
        lex = np.array(self.bm25.get_scores(_tokenize(query_text)), dtype=np.float32)

        sem_n, lex_n = _minmax(sem), _minmax(lex)
        blended = 0.6 * sem_n + 0.4 * lex_n

        k = min(k, len(self.candidates))
        top_idx = np.argpartition(-blended, k - 1)[:k]
        top_idx = top_idx[np.argsort(-blended[top_idx])]
        # carry the normalised semantic score forward as the 'semantic_fit' sub-score
        return [(int(i), float(sem_n[i])) for i in top_idx]


def _build_rationale(c: Candidate, components: dict[str, ScoreComponent],
                     extras: dict, req: JobRequirements) -> str:
    """Assemble a 'why ranked here' sentence from dominant contributors + flags."""
    # top 2 positive contributors
    ordered = sorted(components.values(), key=lambda x: x.contribution, reverse=True)
    strengths = [c2 for c2 in ordered if c2.score >= 0.65][:2]
    parts: list[str] = []
    if strengths:
        parts.append("Strong " + " & ".join(s.name for s in strengths))

    matched = extras.get("matched_skills", [])
    missing = extras.get("missing_required_skills", [])
    if req.required_skills:
        parts.append(f"{len(matched) - len(extras.get('bonus_skills', []))}"
                     f"/{len(req.required_skills)} required skills matched")

    # flags / weaknesses
    flags: list[str] = []
    if missing:
        flags.append("missing " + ", ".join(missing[:3]))
    exp = components.get("experience_fit")
    if exp and exp.score < 0.5:
        flags.append(exp.detail)
    act = components.get("activity")
    if act and act.score < 0.4:
        flags.append("low engagement")
    if c.job_hopping:
        flags.append(f"short avg tenure ({c.avg_tenure_months:.0f}mo)")

    sentence = "; ".join(parts) if parts else "Moderate overall fit"
    if flags:
        sentence += ". Watch-outs: " + "; ".join(flags[:3])
    return sentence + "."


def _normalise_weights(w: SignalWeights) -> dict[str, float]:
    raw = {
        "semantic_fit": w.semantic_fit,
        "skill_match": w.skill_match,
        "experience_fit": w.experience_fit,
        "activity": w.activity,
        "profile_fit": w.profile_fit,
    }
    total = sum(max(0.0, v) for v in raw.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in raw.items()}


def rank(index: CandidateIndex, jd_text: str, weights: SignalWeights | None = None,
         top_n: int = 10, job_id: str | None = None) -> RankResponse:
    """Full pipeline: parse JD -> retrieve -> signal rerank -> explain."""
    t0 = time.perf_counter()
    weights = weights or SignalWeights()
    wnorm = _normalise_weights(weights)
    req = parse_jd(jd_text)

    retrieved = index.retrieve(req)

    results: list[RankedResult] = []
    for idx, sem_score in retrieved:
        cand = index.candidates[idx]

        sk_score, sk_detail, extras = signals.skill_match(cand, req)
        exp_score, exp_detail = signals.experience_fit(cand, req)
        act_score, act_detail = signals.activity_score(cand)
        prof_score, prof_detail = signals.profile_fit(cand, req)

        comps_raw = {
            "semantic_fit": (sem_score, "semantic relevance to JD"),
            "skill_match": (sk_score, sk_detail),
            "experience_fit": (exp_score, exp_detail),
            "activity": (act_score, act_detail),
            "profile_fit": (prof_score, prof_detail),
        }
        components: dict[str, ScoreComponent] = {}
        composite01 = 0.0
        for name, (score, detail) in comps_raw.items():
            w = wnorm[name]
            contrib = score * w
            composite01 += contrib
            components[name] = ScoreComponent(
                name=name, score=round(score, 4), weight=round(w, 4),
                contribution=round(contrib, 4), detail=detail,
            )

        composite = composite01 * 100.0
        fb = feedback_store.boost_for(job_id, cand.id)
        composite = max(0.0, min(100.0, composite + fb))

        results.append(RankedResult(
            candidate=cand,
            rank=0,  # set after sort
            composite_score=round(composite, 2),
            components=list(components.values()),
            matched_skills=extras["matched_skills"],
            missing_required_skills=extras["missing_required_skills"],
            bonus_skills=extras["bonus_skills"],
            feedback_adjustment=round(fb, 2),
            rationale=_build_rationale(cand, components, extras, req),
        ))

    results.sort(key=lambda r: r.composite_score, reverse=True)
    results = results[:top_n]
    for i, r in enumerate(results, start=1):
        r.rank = i

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return RankResponse(
        requirements=req,
        results=results,
        total_candidates=len(index.candidates),
        retrieved=len(retrieved),
        elapsed_ms=round(elapsed_ms, 2),
        weights=weights,
    )
