#!/usr/bin/env python3
"""Repo-root entrypoint for the official reproduce command:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Delegates to the challenge package. Offline, CPU-only, ≤5 min on 100k.
"""
from challenge.rank import main

if __name__ == "__main__":
    main()
