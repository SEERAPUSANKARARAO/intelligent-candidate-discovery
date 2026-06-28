"""Embedding layer — thin, swappable interface around sentence-transformers.

Kept deliberately small so the vector backend can be swapped (e.g. FAISS/Chroma)
without touching the ranking code. For PoC scale (hundreds of candidates) an
in-memory NumPy cosine over a precomputed matrix is genuinely sub-millisecond.

Includes a deterministic hashing-based fallback so the PoC still runs fully
offline if the transformer model cannot be downloaded — semantic quality drops,
but nothing breaks during a demo.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_FALLBACK_DIM = 384

_model = None
_using_fallback = False


def _load_model():
    global _model, _using_fallback
    if _model is not None or _using_fallback:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embeddings] loaded sentence-transformers model '{MODEL_NAME}'")
    except Exception as exc:  # pragma: no cover - environment dependent
        _using_fallback = True
        print(f"[embeddings] WARNING: could not load '{MODEL_NAME}' ({exc}). "
              f"Falling back to deterministic hashing embeddings (reduced quality).")
    return _model


_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Cheap bag-of-hashed-tokens embedding. Deterministic, offline, no deps."""
    vecs = np.zeros((len(texts), _FALLBACK_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vecs[i, h % _FALLBACK_DIM] += 1.0
    return vecs


def is_fallback() -> bool:
    return _using_fallback


def encode(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, _FALLBACK_DIM), dtype=np.float32)
    model = _load_model()
    if model is not None:
        vecs = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype(np.float32)
    # fallback path
    vecs = _hash_embed(texts)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of a single (already-normalised) query vs a normalised matrix.

    Returns a 1-D array of length matrix.shape[0]. Since inputs are L2-normalised,
    cosine reduces to a dot product.
    """
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query
