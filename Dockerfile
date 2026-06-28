# HuggingFace Space (Docker SDK) — serves the FastAPI product UI on port 7860.
# (For the offline Stage-3 ranking image, use Dockerfile.rank instead.)
#
#   docker build -t redrob-ui . && docker run --rm -p 7860:7860 redrob-ui
#
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DEMO_DATA=/app/data/demo_candidates.jsonl

# Lean deps for the FastAPI UI (no torch/streamlit needed for the product UI)
RUN pip install --no-cache-dir numpy scipy scikit-learn fastapi "uvicorn[standard]" pydantic

COPY challenge/ challenge/
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/demo_candidates.jsonl data/sample_candidates.jsonl data/

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
