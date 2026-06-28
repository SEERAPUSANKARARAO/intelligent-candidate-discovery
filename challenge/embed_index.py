"""Offline pre-computation of BGE embeddings + a FAISS index.

The rules ALLOW pre-computation outside the 5-minute ranking window, so we encode
the whole pool once here and persist artifacts the ranking step loads instantly:

  artifacts/emb.npy     float32 (N, d) L2-normalised candidate embeddings
  artifacts/ids.json    candidate_id per row
  artifacts/jd.npy      the JD query embedding (so rank-time needs NO torch/model)
  artifacts/faiss.index FAISS inner-product index (dense recall / vector DB demo)
  artifacts/meta.json   model name + dim

Because the JD vector is cached too, the ranking step computes semantic scores as a
plain dot product — fully offline, CPU-only, no model load.

    python -m challenge.embed_index --candidates data/candidates.jsonl --out challenge/artifacts
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import retrieval
from .schema import load_candidates

MODEL = "BAAI/bge-small-en-v1.5"
BGE_LOCAL = Path(__file__).resolve().parent / "models" / "bge-small"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(BGE_LOCAL) if BGE_LOCAL.exists() else MODEL)


def precompute(candidates_path: str, out_dir: str, batch_size: int = 64) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_candidates(candidates_path)
    docs = [retrieval.candidate_document(r) for r in records]
    ids = [r["candidate_id"] for r in records]

    model = _model()
    print(f"[embed] encoding {len(docs):,} candidates with {MODEL} (CPU) ...")
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True, batch_size=batch_size).astype(np.float32)
    jd_emb = model.encode([QUERY_PREFIX + retrieval.jd_query_text()],
                          normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    np.save(out / "emb.npy", emb)
    np.save(out / "jd.npy", jd_emb)
    (out / "ids.json").write_text(json.dumps(ids))
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "dim": int(emb.shape[1]),
                                               "n": len(ids)}))
    try:
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(out / "faiss.index"))
        print(f"[embed] FAISS index built ({index.ntotal} vectors)")
    except Exception as e:  # pragma: no cover
        print(f"[embed] faiss skipped ({e})")
    print(f"[embed] wrote artifacts to {out}")
    return out


class PrecomputedEmbeddings:
    """Rank-time loader: semantic scores via cached dot product (no torch needed)."""

    def __init__(self, art_dir: str):
        art = Path(art_dir)
        self.emb = np.load(art / "emb.npy")
        self.jd = np.load(art / "jd.npy")
        self.ids = json.loads((art / "ids.json").read_text())
        self.row = {cid: i for i, cid in enumerate(self.ids)}
        self.scores = self.emb @ self.jd  # (N,) cosine (already normalised)

    def score_for(self, candidate_id: str) -> float:
        i = self.row.get(candidate_id)
        return float(self.scores[i]) if i is not None else 0.0

    def dense_recall(self, k: int) -> list[str]:
        """Top-k candidate_ids by embedding similarity to the JD (FAISS-style)."""
        idx = np.argsort(-self.scores)[:k]
        return [self.ids[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--out", default="challenge/artifacts")
    args = ap.parse_args()
    precompute(args.candidates, args.out)


if __name__ == "__main__":
    main()
