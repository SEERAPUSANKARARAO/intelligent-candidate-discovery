"""Streamlit sandbox — a spec-approved hosted demo (Streamlit Cloud / HF Spaces).

Runs the real offline engine on a small pool and shows the ranked shortlist with
evidence-cited reasoning, sub-scores, and honeypot flags. The FastAPI app
(`backend/main.py`) is the richer product UI; this is the lightweight sandbox.

    streamlit run challenge/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from backend.service import RankerService, jd_info
from challenge.synth import generate

st.set_page_config(page_title="Redrob · Candidate Discovery", page_icon="🎯", layout="wide")

POOL = Path("data/demo_candidates.jsonl")
if not POOL.exists():
    POOL = Path("data/sample_candidates.jsonl")
if not POOL.exists():
    generate(800, "data/synth"); POOL = Path("data/synth/candidates.jsonl")


@st.cache_resource(show_spinner="Loading engine + candidate pool…")
def _svc(path: str):
    return RankerService(path)


svc = _svc(str(POOL))
jd = jd_info()

st.title("🎯 Intelligent Candidate Discovery")
st.caption("Redrob AI · India Runs Track-1 — offline, honeypot-immune, explainable ranking.")

with st.sidebar:
    st.header("⚙️ Controls")
    top_n = st.slider("Shortlist size", 5, 100, 20)
    st.subheader("Signal weights")
    w = {
        "semantic": st.slider("Semantic", 0.0, 1.0, 0.20, 0.01),
        "skill": st.slider("Skill", 0.0, 1.0, 0.25, 0.01),
        "career": st.slider("Career", 0.0, 1.0, 0.33, 0.01),
        "experience": st.slider("Experience", 0.0, 1.0, 0.22, 0.01),
    }
    use_sem = st.checkbox("Semantic rerank", True)
    use_beh = st.checkbox("Behavioural signals", True)
    use_hp = st.checkbox("Honeypot defence", True)
    search = st.text_input("Search", "")
    with st.expander("Target role (JD)"):
        st.markdown(f"**{jd['title']}** — {jd['company']}")
        st.write(jd["text"])

tot = sum(max(0, v) for v in w.values()) or 1.0
w = {k: max(0, v) / tot for k, v in w.items()}
cards = svc.rank(w, use_sem, use_beh, use_hp, top_n, search)
stats = svc.stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pool", f"{stats['pool_size']:,}")
c2.metric("Strong", stats["strong_candidates"])
c3.metric("Honeypots caught", stats["honeypots_detected"])
c4.metric("Engine", stats["backend"].split("(")[0])

st.subheader(f"🏆 Top {len(cards)}")
df = pd.DataFrame([{
    "rank": c["rank"], "candidate_id": c["candidate_id"], "name": c["name"],
    "title": c["title"], "company": c["company"], "yoe": c["yoe"],
    "score": c["score"], "honeypot": "⚠" if c["is_honeypot"] else "",
    "reasoning": c["reasoning"],
} for c in cards])
st.dataframe(df, use_container_width=True, hide_index=True,
             column_config={"reasoning": st.column_config.TextColumn(width="large")})

pick = st.selectbox("Inspect candidate", [c["candidate_id"] for c in cards])
if pick:
    d = svc.candidate(pick)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{d['name']}** · {d['title']}")
        st.caption(f"🏢 {d['company']} · {d['industry']} · ⏳ {d['yoe']}y · 📍 {d['location']}")
        if d["is_honeypot"]:
            st.error("HONEYPOT — " + "; ".join(d["honeypot_reasons"]))
        else:
            st.success(d["reasoning"])
        st.markdown("**Skills**")
        st.dataframe(pd.DataFrame(d["skills"]), hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Career**")
        st.dataframe(pd.DataFrame(d["career"])[["title", "company", "duration_months", "is_current"]],
                     hide_index=True, use_container_width=True)
        st.markdown("**Behavioural signals (23)**")
        st.json(d["signals"], expanded=False)

with st.expander("🛡️ Traps caught & excluded"):
    st.dataframe(pd.DataFrame(svc.traps()), hide_index=True, use_container_width=True)
