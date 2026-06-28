"""Tests for the India Runs Track-1 ranker (real-schema engine)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from challenge import features, honeypot, metrics, reasoning, scoring
from challenge.rank import RankConfig, load_pool, run_pipeline, write_submission
from challenge.synth import generate, load_ground_truth
from challenge.validate_submission import validate


# --- metrics ---------------------------------------------------------------------
def test_ndcg_and_composite():
    rel = {"a": 5, "b": 3, "c": 1, "d": 0}
    assert metrics.ndcg_at_k(["a", "b", "c", "d"], rel, 4) == pytest.approx(1.0)
    assert 0 <= metrics.composite(["a", "b", "c", "d"], rel) <= 1.0
    assert metrics.composite(["d", "c", "b", "a"], rel) < metrics.composite(["a", "b", "c", "d"], rel)


def test_honeypot_rate_dq():
    rep = metrics.full_report(["h1", "ok", "h2"], {"ok": 4}, {"h1", "h2"}, top_k=3)
    assert rep["honeypots_in_top_k"] == 2 and rep["disqualified"]


# --- token matching: no short-substring false positives --------------------------
def test_token_matching_avoids_substring_traps():
    toks = features._tok("retail email storage average planning")
    assert not features._present("ai", toks, "retail email")   # not inside 'retail'/'email'
    assert not features._present("rag", toks, "storage average")
    assert not features._present("ann", toks, "planning")
    toks2 = features._tok("faiss bm25 ai rag")
    assert features._present("ai", toks2, "faiss bm25 ai rag")
    assert features._present("rag", toks2, "faiss bm25 ai rag")


def test_skill_trust_discounts_fabrication():
    real = {"proficiency": "expert", "endorsements": 30, "duration_months": 40, "assessment_score": 0.9}
    fake = {"proficiency": "expert", "endorsements": 0, "duration_months": 0}
    assert features.skill_trust(real) > 0.7
    assert features.skill_trust(real) > 3 * features.skill_trust(fake)


# --- honeypot detection (date/duration impossibility) ----------------------------
def test_honeypot_detected():
    rec = {"candidate_id": "CAND_0000001", "years_of_experience": 3,
           "skills": [{"name": n, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
                      for n in ["faiss", "bm25", "rag", "bert"]],
           "career": [{"title": "AI Engineer", "company": "X", "start_date": "2023-01-01",
                       "end_date": None, "duration_months": 96}]}
    flagged, reasons = honeypot.detect(rec)
    assert flagged and reasons


# --- keyword-stuffer vs genuine via fusion ---------------------------------------
def test_keyword_stuffer_scores_below_genuine():
    stuffer = {"candidate_id": "CAND_0000002", "title": "Marketing Manager", "headline": "",
               "summary": "Marketing campaigns and branding.", "location": "Pune", "country": "India",
               "years_of_experience": 6, "company": "Globex", "industry": "Manufacturing",
               "skills": [{"name": s, "proficiency": "expert", "endorsements": 1, "duration_months": 2}
                          for s in ["FAISS", "RAG", "BM25", "Embeddings", "LLMs"]],
               "career": [{"title": "Marketing Manager", "company": "Globex", "industry": "Manufacturing",
                           "start_date": "2019-01-01", "end_date": None, "duration_months": 72,
                           "is_current": True, "description": "Ran marketing campaigns."}],
               "education": [], "signals": {}}
    genuine = {"candidate_id": "CAND_0000003", "title": "Senior ML Engineer", "headline": "",
               "summary": "Built production retrieval and ranking with embeddings and FAISS.",
               "location": "Bangalore", "country": "India", "years_of_experience": 7,
               "company": "Flipkart", "industry": "E-commerce",
               "skills": [{"name": s, "proficiency": "expert", "endorsements": 25, "duration_months": 36}
                          for s in ["FAISS", "BM25", "PyTorch", "Embeddings"]],
               "career": [{"title": "Senior ML Engineer", "company": "Flipkart", "industry": "E-commerce",
                           "start_date": "2019-01-01", "end_date": None, "duration_months": 72,
                           "is_current": True,
                           "description": "Shipped production semantic search with FAISS, reranking, NDCG eval."}],
               "education": [], "signals": {"open_to_work": True, "last_active_days": 3,
                                            "recruiter_response_rate": 0.8}}
    fs, fg = features.extract(stuffer), features.extract(genuine)
    s_stuff = scoring.fuse(stuffer, fs, 0.5)["final_score"]
    s_gen = scoring.fuse(genuine, fg, 0.5)["final_score"]
    assert s_gen > s_stuff


# --- end-to-end ------------------------------------------------------------------
def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        cand, gt_path = generate(2500, d, seed=7)
        gt = load_ground_truth(gt_path)
        honeypots = {c for c, v in gt.items() if v["is_honeypot"]}
        relevance = {c: v["relevance"] for c, v in gt.items()}
        records, documents, bm25 = load_pool(cand)
        results, _ = run_pipeline(records, documents, bm25, RankConfig(top_k=100))
        ranked = [r["candidate_id"] for r in results]

        rep = metrics.full_report(ranked, relevance, honeypots, 100)
        assert not rep["disqualified"]
        assert rep["honeypots_in_top_k"] == 0
        assert rep["composite"] > 0.4

        # reasoning has no skill hallucination on the top rows
        by_id = {r["candidate_id"]: r for r in records}
        for res in results[:20]:
            assert reasoning.verify_no_hallucination(res, by_id[res["candidate_id"]])

        out = write_submission(results, Path(d) / "team_test.csv")
        assert validate(out, cand) == []
