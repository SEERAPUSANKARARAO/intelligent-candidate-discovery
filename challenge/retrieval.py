"""Offline retrieval: hybrid BM25 (lexical) + TF-IDF / embedding (semantic).

All CPU-only, no network, no model downloads — exactly the contest constraint.
BM25 is implemented over a sparse term-document matrix so it scales to 100k rows
in a couple of seconds. The semantic channel defaults to TF-IDF cosine (word +
character n-grams, which already sees light paraphrase) and *transparently
upgrades* to sentence-transformers if a local model happens to be available.

The funnel uses these in two places:
  Stage 1  — BM25 over ALL candidates -> cheap recall shortlist.
  Stage 3  — semantic similarity over the SHORTLIST only -> precise rerank signal.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import jd

_TOKEN_RE = re.compile(r"[a-z0-9+#./@-]+")

# vendored BGE weights live here so the judge environment never hits the network.
BGE_LOCAL_PATH = Path(__file__).resolve().parent / "models" / "bge-small"
BGE_HUB_NAME = "BAAI/bge-small-en-v1.5"
# BGE retrieval expects this instruction prefix on the QUERY only (not documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def candidate_document(rec: dict) -> str:
    """Flatten a canonical candidate into one searchable text blob."""
    skills = " ".join(s.get("name", "") for s in rec.get("skills", []))
    career = rec.get("career", [])
    titles = " ".join(j.get("title", "") for j in career)
    descs = " ".join(j.get("description", "") for j in career)
    companies = " ".join(j.get("company", "") for j in career)
    head = f"{rec.get('title','')} {rec.get('headline','')} {rec.get('summary','')}"
    return f"{head}. {titles}. {skills}. {descs}. {companies}"


def jd_query_text() -> str:
    surface = [s for v in jd.MUST_HAVE.values() for s in v]
    return jd.JD_TEXT + " " + " ".join(surface) + " " + " ".join(jd.IDEAL_TITLES)


class BM25Index:
    """Okapi BM25 over a sparse count matrix. Built once over the full pool."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)[a-z0-9+#./@-]+", lowercase=True)
        self.tf = self.vectorizer.fit_transform(documents).astype(np.float32)  # (N, V)
        n_docs = self.tf.shape[0]
        df = np.asarray((self.tf > 0).sum(axis=0)).ravel()                     # (V,)
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 0.0

    def get_scores(self, query: str) -> np.ndarray:
        """BM25 score for every document against ``query``. Returns (N,)."""
        q_idx = [self.vectorizer.vocabulary_[t] for t in set(tokenize(query))
                 if t in self.vectorizer.vocabulary_]
        if not q_idx:
            return np.zeros(self.tf.shape[0], dtype=np.float32)
        denom_len = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        scores = np.zeros(self.tf.shape[0], dtype=np.float32)
        tf_csc = self.tf.tocsc()
        for j in q_idx:
            col = tf_csc.getcol(j).toarray().ravel()           # tf of term j per doc
            scores += self.idf[j] * (col * (self.k1 + 1.0)) / (col + denom_len)
        return scores


# --- semantic channel ------------------------------------------------------------
_st_model = None
_st_tried = False


def _backend_pref() -> str:
    """tfidf (default, fast/torch-free) | bge — set via SEMANTIC_BACKEND env var.

    TF-IDF is the reliable default so the pipeline runs with no torch dependency
    and stays well inside the time budget. BGE is an explicit, measured upgrade."""
    return os.environ.get("SEMANTIC_BACKEND", "tfidf").lower()


def _try_sentence_transformer():
    """Load BGE from the vendored offline copy when SEMANTIC_BACKEND=bge.

    Returns None (→ TF-IDF) unless BGE is explicitly requested AND importable."""
    global _st_model, _st_tried
    if _backend_pref() != "bge":
        return None
    if _st_tried:
        return _st_model
    _st_tried = True
    try:  # pragma: no cover - environment dependent
        from sentence_transformers import SentenceTransformer
        if BGE_LOCAL_PATH.exists():
            # vendored: force offline so the judge box never reaches the network
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _st_model = SentenceTransformer(str(BGE_LOCAL_PATH))
        else:
            _st_model = SentenceTransformer(BGE_HUB_NAME)  # first run only
    except Exception:
        _st_model = None
    return _st_model


def semantic_scores(documents: list[str], query: str) -> tuple[np.ndarray, str]:
    """Cosine similarity of every doc vs the query. Returns (scores, backend_name).

    Uses vendored BGE embeddings if present (query gets the BGE instruction
    prefix; documents do not); otherwise TF-IDF cosine over word + bigram tokens.
    Both are fully offline."""
    if not documents:
        return np.zeros(0, dtype=np.float32), "none"
    model = _try_sentence_transformer()
    if model is not None:  # pragma: no cover - needs vendored weights
        doc_emb = model.encode(documents, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True,
                               batch_size=64)
        q_emb = model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)[0]
        return (doc_emb @ q_emb).astype(np.float32), "bge-small-en-v1.5"

    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1,
                          token_pattern=r"(?u)[a-z0-9+#./@-]+")
    mat = vec.fit_transform(documents + [query])
    doc_mat, q_vec = mat[:-1], mat[-1]
    sims = (doc_mat @ q_vec.T).toarray().ravel()
    return sims.astype(np.float32), "tfidf-cosine"


def minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
