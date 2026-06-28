"""Submission validator — mirrors the OFFICIAL rules (submission_spec §2-3).

Header EXACTLY `candidate_id,rank,score,reasoning`; exactly 100 data rows; ranks
1-100 each once; CAND_XXXXXXX ids unique & present in the pool; score
non-increasing by rank; equal scores ⇒ candidate_id ascending.

    python -m challenge.validate_submission submission.csv [--pool data/candidates.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

EXPECTED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CAND_RE = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_ROWS = 100


def validate(path: str | Path, pool_path: str | None = None) -> list[str]:
    errors: list[str] = []
    path = Path(path)
    if path.suffix.lower() != ".csv":
        errors.append("filename must use a .csv extension")

    rows = list(csv.reader(open(path, encoding="utf-8", newline="")))
    if not rows:
        return ["empty file"]
    if rows[0] != EXPECTED_HEADER:
        errors.append(f"header must be {EXPECTED_HEADER}, found {rows[0]}")
    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    if len(body) != EXPECTED_ROWS:
        errors.append(f"must have exactly {EXPECTED_ROWS} data rows, found {len(body)}")

    pool_ids = None
    if pool_path:
        from .schema import iter_raw
        pool_ids = {c.get("candidate_id") for c in iter_raw(pool_path)}

    seen_ids, seen_ranks, by_rank = set(), set(), []
    for i, row in enumerate(body, start=2):
        if len(row) != 4:
            errors.append(f"row {i}: expected 4 columns, got {len(row)}")
            continue
        cid, rank_s, score_s, _ = row
        cid = cid.strip()
        if not CAND_RE.match(cid):
            errors.append(f"row {i}: candidate_id '{cid}' must be CAND_XXXXXXX")
        elif cid in seen_ids:
            errors.append(f"row {i}: duplicate candidate_id {cid}")
        else:
            seen_ids.add(cid)
        if pool_ids is not None and cid not in pool_ids:
            errors.append(f"row {i}: candidate_id {cid} not in pool")
        try:
            rank = int(rank_s)
            if str(rank) != rank_s or not 1 <= rank <= 100:
                raise ValueError
            if rank in seen_ranks:
                errors.append(f"row {i}: duplicate rank {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"row {i}: rank must be integer 1-100, got '{rank_s}'")
            rank = None
        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"row {i}: score '{score_s}' not a float")
            score = None
        if rank is not None and score is not None:
            by_rank.append((rank, score, cid))

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"missing ranks: {sorted(missing)}")
    by_rank.sort(key=lambda x: x[0])
    for a, b in zip(by_rank, by_rank[1:]):
        if a[1] < b[1]:
            errors.append(f"score must be non-increasing: rank {a[0]} ({a[1]}) < rank {b[0]} ({b[1]})")
        if a[1] == b[1] and a[2] > b[2]:
            errors.append(f"equal scores at ranks {a[0]},{b[0]} need candidate_id ascending")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pool", default=None)
    args = ap.parse_args()
    errs = validate(args.path, args.pool)
    if errs:
        print(f"INVALID — {len(errs)} issue(s):")
        for e in errs[:25]:
            print(f"  - {e}")
        sys.exit(1)
    print(f"VALID — {args.path} passes all format checks.")


if __name__ == "__main__":
    main()
