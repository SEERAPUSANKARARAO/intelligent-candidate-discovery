"""Optional cross-encoder reranking of the fusion top-N.

A bi-encoder (BGE) scores query↔doc independently; a cross-encoder reads the pair
jointly and is more accurate at the very top — which is exactly where NDCG@10 is
won. Run it only on the top-N (default 120) so it stays inside the CPU budget.

Offline: loads a vendored cross-encoder from challenge/models/cross-encoder, or
downloads once (then works offline). Opt-in via rank/evaluate flags; the default
pipeline does not require torch.

Vendor once:
    python -c "from sentence_transformers import CrossEncoder as C; \
        C('cross-encoder/ms-marco-MiniLM-L-6-v2').save('challenge/models/cross-encoder')"
"""

from __future__ import annotations

import os
from pathlib import Path

from . import jd

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL = Path(__file__).resolve().parent / "models" / "cross-encoder"
_ce = None


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    global _ce
    if _ce is not None:
        return _ce
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import CrossEncoder
    _ce = CrossEncoder(str(LOCAL) if LOCAL.exists() else MODEL)
    return _ce


def rerank(results: list[dict], texts: dict[str, str], top_n: int = 120,
           weight: float = 0.5) -> list[dict]:
    """Blend a cross-encoder score into the top-N final scores, then re-sort.

    ``results`` are fusion outputs (with final_score, candidate_id); ``texts`` maps
    candidate_id -> document text. Honeypots (already floored) are left untouched."""
    model = _load()
    head = [r for r in results[:top_n] if not r.get("is_honeypot")]
    if not head:
        return results
    query = jd.JD_TEXT
    pairs = [(query, texts.get(r["candidate_id"], "")) for r in head]
    ce = model.predict(pairs, show_progress_bar=False)
    # normalise CE to [0,1]
    lo, hi = float(min(ce)), float(max(ce))
    span = (hi - lo) or 1.0
    for r, s in zip(head, ce):
        ce_n = (float(s) - lo) / span
        r["ce_score"] = round(ce_n, 4)
        r["final_score"] = (1 - weight) * r["final_score"] + weight * ce_n
    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    return results
