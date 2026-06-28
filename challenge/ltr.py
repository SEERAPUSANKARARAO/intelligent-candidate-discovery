"""Learning-to-rank (LambdaMART via LightGBM) over the engineered features.

LambdaMART directly optimises NDCG — the grader's headline metric. We train it on
the labelled synthetic tiers (one JD ⇒ a single ranking group) and ship the booster
as an offline artifact. Because it consumes the SAME interpretable features as the
hand-tuned fusion, it generalises better than a black-box text model and stays
explainable. Used as an optional ensemble with the fusion score (honeypot floor
always applied downstream).

    python -m challenge.ltr --train data/synth/candidates.jsonl --gt data/synth/ground_truth.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import features as featmod
from .schema import load_candidates
from .synth import load_ground_truth

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "ltr.txt"
FEATS = featmod.LTR_FEATURES


def _vec(f: dict) -> list[float]:
    return [float(f.get(k, 0) or 0) for k in FEATS]


def train(candidates_path: str, gt_path: str, out: str | Path = ARTIFACT) -> Path:
    import lightgbm as lgb
    records = load_candidates(candidates_path)
    gt = load_ground_truth(gt_path)
    X, y = [], []
    for r in records:
        X.append(_vec(featmod.extract(r)))
        y.append(gt.get(r["candidate_id"], {}).get("relevance", 0))
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
    ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg",
                            n_estimators=300, learning_rate=0.05, num_leaves=31,
                            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbose=-1)
    ranker.fit(X, y, group=[len(X)])  # one JD = one ranking group
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(out))
    imp = sorted(zip(FEATS, ranker.feature_importances_), key=lambda x: -x[1])
    print(f"[ltr] trained on {len(X)} candidates -> {out}")
    print("[ltr] top features:", ", ".join(f"{k}({v})" for k, v in imp[:8]))
    return Path(out)


class LTRModel:
    def __init__(self, path: str | Path = ARTIFACT):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(path))

    def score_features(self, feats: list[dict]) -> np.ndarray:
        X = np.array([_vec(f) for f in feats], dtype=np.float32)
        return self.booster.predict(X)

    @staticmethod
    def exists(path: str | Path = ARTIFACT) -> bool:
        return Path(path).exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synth/candidates.jsonl")
    ap.add_argument("--gt", default="data/synth/ground_truth.csv")
    ap.add_argument("--out", default=str(ARTIFACT))
    args = ap.parse_args()
    train(args.train, args.gt, args.out)


if __name__ == "__main__":
    main()
