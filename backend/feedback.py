"""Recruiter feedback loop — lightweight personalisation.

Thumbs up/down on a candidate (for a given job) are persisted to a JSON file.
On subsequent ranks we apply a small per-(job, candidate) boost/penalty so the
shortlist adapts to the recruiter's taste. This demonstrates a learn-to-rank-lite
behaviour without a training pipeline; the README notes the production path.

Boost is expressed in composite *points* (the composite is shown on a 0–100
scale), so a thumbs-up nudges a candidate up by a few points and similar profiles
get a fractional, decaying nudge.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEEDBACK_PATH = DATA_DIR / "feedback.json"

# composite-point boost per net vote, capped so feedback tunes rather than dominates
POINTS_PER_VOTE = 4.0
MAX_BOOST = 12.0


class FeedbackStore:
    def __init__(self, path: Path = FEEDBACK_PATH):
        self.path = path
        # { job_id: { candidate_id: net_votes } }
        self._votes: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._votes = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._votes = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._votes, indent=2))

    def record(self, job_id: str, candidate_id: str, vote: int) -> int:
        """Record a vote (+1/-1). Returns the new net vote tally for that pair."""
        vote = 1 if vote > 0 else -1
        job = self._votes.setdefault(job_id, {})
        job[candidate_id] = job.get(candidate_id, 0) + vote
        self._save()
        return job[candidate_id]

    def boost_for(self, job_id: str | None, candidate_id: str) -> float:
        """Composite-point adjustment for a candidate under a given job."""
        if not job_id:
            return 0.0
        net = self._votes.get(job_id, {}).get(candidate_id, 0)
        return max(-MAX_BOOST, min(MAX_BOOST, net * POINTS_PER_VOTE))

    def upvoted_ids(self, job_id: str | None) -> set[str]:
        if not job_id:
            return set()
        return {cid for cid, net in self._votes.get(job_id, {}).items() if net > 0}

    def reset(self, job_id: str | None = None) -> None:
        if job_id is None:
            self._votes = {}
        else:
            self._votes.pop(job_id, None)
        self._save()


# module-level singleton used by the API
store = FeedbackStore()
