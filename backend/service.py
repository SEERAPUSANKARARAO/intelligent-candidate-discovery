"""In-process ranking service for the demo/sandbox UI.

Wraps the offline `challenge` engine. Loads a candidate pool once, precomputes
features + hybrid (BM25 ⊕ semantic) scores + honeypot flags, then re-fuses on every
request so weight-slider / toggle changes feel instant. This is the same engine
that produces the official submission — just driven interactively over a small pool.
"""

from __future__ import annotations

from pathlib import Path

from challenge import features as featmod
from challenge import honeypot, jd, reasoning, retrieval, scoring
from challenge.schema import load_candidates


class RankerService:
    def __init__(self, pool_path: str | Path):
        self.records = load_candidates(pool_path)
        self.docs = [retrieval.candidate_document(r) for r in self.records]
        self.bm25 = retrieval.BM25Index(self.docs)
        self.feats = [featmod.extract(r) for r in self.records]
        self.flags = [honeypot.detect(r) for r in self.records]
        self.by_id = {r["candidate_id"]: i for i, r in enumerate(self.records)}
        q = retrieval.jd_query_text()
        sem, self.backend = retrieval.semantic_scores(self.docs, q)
        self.hybrid = retrieval.minmax(0.6 * retrieval.minmax(sem)
                                       + 0.4 * retrieval.minmax(self.bm25.get_scores(q)))

    # ------------------------------------------------------------------ ranking
    def rank(self, weights, use_semantic=True, use_behavioral=True, use_honeypot=True,
             top_n=20, search="") -> list[dict]:
        q = (search or "").strip().lower()
        idxs = range(len(self.records))
        if q:
            idxs = [i for i in idxs if q in (self.records[i]["name"] + " "
                    + self.records[i]["title"] + " "
                    + " ".join(s["name"] for s in self.records[i]["skills"])).lower()]
        scored = []
        for i in idxs:
            sem = float(self.hybrid[i]) if use_semantic else 0.0
            s = scoring.fuse(self.records[i], self.feats[i], sem, weights,
                             use_semantic, use_behavioral, use_honeypot)
            scored.append(s)
        top_score = max((s["final_score"] for s in scored), default=1.0) or 1.0
        for s in scored:
            s["final_score"] = round(s["final_score"] / top_score, 6)
        scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
        out = scored[:top_n]
        for rank, s in enumerate(out, start=1):
            s["rank"] = rank
            s["reasoning"] = reasoning.build(s, rank, top_n)
        return [self._card(s) for s in out]

    def _card(self, s: dict) -> dict:
        i = self.by_id[s["candidate_id"]]
        r = self.records[i]
        f = s["features"]
        return {
            "rank": s["rank"], "candidate_id": s["candidate_id"], "name": r["name"],
            "title": r["title"], "company": r["company"], "industry": r["industry"],
            "location": r["location"], "yoe": r["years_of_experience"],
            "score": round(s["final_score"], 4),
            "sub": {"semantic": s["semantic"], "skill": s["skill"], "career": s["career"],
                    "experience": s["experience"]},
            "behavioral_multiplier": s["behavioral_multiplier"],
            "matched_groups": f["matched_groups"], "present_skills": f["facts"]["present_skills"],
            "is_honeypot": s["is_honeypot"], "honeypot_reasons": s["honeypot_reasons"],
            "concerns": f["concerns"], "reasoning": s["reasoning"],
            "product_frac": f["product_frac"], "services_frac": f["services_frac"],
            "availability": f["availability"], "applied_ml_frac": f["applied_ml_frac"],
        }

    # ------------------------------------------------------------------ detail
    def candidate(self, cid: str) -> dict | None:
        i = self.by_id.get(cid)
        if i is None:
            return None
        r = self.records[i]
        f = self.feats[i]
        hp, reasons = self.flags[i]
        s = scoring.fuse(r, f, float(self.hybrid[i]), dict(scoring.DEFAULT_WEIGHTS))
        return {
            "candidate_id": cid, "name": r["name"], "title": r["title"],
            "headline": r["headline"], "summary": r["summary"], "company": r["company"],
            "industry": r["industry"], "company_size": r["company_size"],
            "location": r["location"], "country": r["country"],
            "yoe": r["years_of_experience"], "education": r["education"],
            "skills": r["skills"], "career": r["career"], "signals": r["signals"],
            "features": {k: f[k] for k in (
                "skill_score", "career_score", "experience_score", "behavioral_multiplier",
                "title_fit", "services_frac", "product_frac", "applied_ml_frac",
                "narrative_hits", "yoe_fit", "matched_groups", "concerns")},
            "is_honeypot": hp, "honeypot_reasons": reasons,
            "reasoning": reasoning.build({**s, "rank": 1}, 1),
        }

    # ------------------------------------------------------------------ meta
    def stats(self) -> dict:
        honeypots = sum(1 for hp, _ in self.flags if hp)
        services = sum(1 for f in self.feats if f["services_frac"] >= 0.6)
        strong = sum(1 for f in self.feats if f["skill_score"] >= 0.5 and f["career_score"] >= 0.5)
        return {"pool_size": len(self.records), "honeypots_detected": honeypots,
                "services_heavy": services, "strong_candidates": strong,
                "backend": self.backend}

    def traps(self, limit=30) -> list[dict]:
        out = []
        for i, (hp, reasons) in enumerate(self.flags):
            if hp:
                r = self.records[i]
                out.append({"candidate_id": r["candidate_id"], "name": r["name"],
                            "title": r["title"], "type": "honeypot",
                            "reasons": reasons[:2]})
        # keyword stuffers: AI groups present but non-AI title + low trust corroboration
        for i, f in enumerate(self.feats):
            if len(out) >= limit:
                break
            r = self.records[i]
            t = r["title"].lower()
            if (not self.flags[i][0] and f["n_matched_groups"] >= 4
                    and f["title_fit"] < 0.3 and f["career_score"] < 0.35):
                out.append({"candidate_id": r["candidate_id"], "name": r["name"],
                            "title": r["title"], "type": "keyword-stuffer",
                            "reasons": [f"{f['n_matched_groups']} AI skill areas but "
                                        f"non-AI title & no corroborating career"]})
        return out[:limit]


def jd_info() -> dict:
    return {"title": jd.JD_TITLE, "company": jd.JD_COMPANY, "text": jd.JD_TEXT,
            "must_have": list(jd.MUST_HAVE.keys()), "nice_to_have": list(jd.NICE_TO_HAVE.keys())}
