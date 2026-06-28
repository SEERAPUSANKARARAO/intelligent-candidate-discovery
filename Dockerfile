# Stage-3 reproduction image: runs the DEFAULT ranking step under the contest
# constraints (CPU-only, offline at runtime, ≤5 min, ≤16 GB). Build once (this is
# the only step that touches the network); run with no network.
#
#   docker build -t redrob-ranker .
#   docker run --rm --network none -m 16g \
#       -v "$PWD/data:/data" redrob-ranker \
#       --candidates /data/candidates.jsonl --out /data/submission.csv
#
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# minimal deps for the default (TF-IDF + BM25 + structured) pipeline
COPY requirements-rank.txt .
RUN pip install --no-cache-dir -r requirements-rank.txt

# engine + entrypoint (no data baked in — mount it at run time)
COPY challenge/ challenge/
COPY rank.py .

ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/data/submission.csv"]
