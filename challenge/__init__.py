"""Redrob India Runs — Track 1 (Data & AI) candidate-ranking submission.

A self-contained, **fully offline, CPU-only** pipeline that ranks the top-100
candidates for the *Senior AI Engineer (Founding Team)* role at Redrob AI from a
100k-row ``candidates.jsonl`` file — explicitly designed to:

  * see semantic fit beyond keywords (hybrid BM25 + vector retrieval),
  * reject the dataset's keyword-stuffed honeypots (trust + corroboration),
  * integrate the 23 behavioural / activity signals, and
  * optimise the scored metric (NDCG@10) with a labelled synthetic harness.

Public entry points:
  * ``challenge.rank``      — CLI: candidates.jsonl -> submission.csv
  * ``challenge.synth``     — synthetic data + graded ground truth + honeypots
  * ``challenge.evaluate``  — NDCG/MRR/MAP/recall + ablation study
  * ``challenge.app``       — Streamlit demo
"""

__all__ = ["jd", "synth", "retrieval", "scoring", "metrics", "rank", "evaluate"]
