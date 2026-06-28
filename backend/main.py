"""FastAPI app — REST API + static UI for the Redrob candidate-ranking product.

Doubles as the required hosted sandbox: loads a small candidate pool and runs the
real offline engine interactively. Set DEMO_DATA to point at a candidate file
(.jsonl/.jsonl.gz); defaults to the curated demo pool.

    uvicorn backend.main:app --port 8000     # then open http://localhost:8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import RankerService, jd_info

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DEFAULT_POOL = os.environ.get("DEMO_DATA", str(ROOT / "data" / "demo_candidates.jsonl"))
if not Path(DEFAULT_POOL).exists():
    DEFAULT_POOL = str(ROOT / "data" / "sample_candidates.jsonl")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] loading pool {DEFAULT_POOL} ...")
    _state["svc"] = RankerService(DEFAULT_POOL)
    print(f"[startup] ready: {_state['svc'].stats()}")
    yield
    _state.clear()


app = FastAPI(title="Redrob Talent Intelligence", version="2.0.0", lifespan=lifespan)


def svc() -> RankerService:
    s = _state.get("svc")
    if s is None:
        raise HTTPException(503, "service not ready")
    return s


class Weights(BaseModel):
    semantic: float = 0.20
    skill: float = 0.25
    career: float = 0.33
    experience: float = 0.22


class RankReq(BaseModel):
    weights: Weights = Weights()
    use_semantic: bool = True
    use_behavioral: bool = True
    use_honeypot: bool = True
    top_n: int = 20
    search: str = ""


@app.get("/api/jd")
def get_jd() -> dict:
    return jd_info()


@app.get("/api/stats")
def get_stats() -> dict:
    return svc().stats()


@app.post("/api/rank")
def post_rank(req: RankReq) -> dict:
    w = req.weights.model_dump()
    total = sum(max(0.0, v) for v in w.values()) or 1.0
    w = {k: max(0.0, v) / total for k, v in w.items()}  # normalise
    cards = svc().rank(w, req.use_semantic, req.use_behavioral, req.use_honeypot,
                       max(1, min(100, req.top_n)), req.search)
    return {"results": cards, "weights": w}


@app.get("/api/candidate/{cid}")
def get_candidate(cid: str) -> dict:
    c = svc().candidate(cid)
    if c is None:
        raise HTTPException(404, "unknown candidate")
    return c


@app.get("/api/traps")
def get_traps() -> dict:
    return {"traps": svc().traps()}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
