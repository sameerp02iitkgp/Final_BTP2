FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# System deps (git-lfs only if you use large files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system || true

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["bash", "-lc", "if [ -f /app/app.py ]; then streamlit run /app/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true; elif [ -f /app/src/app.py ]; then streamlit run /app/src/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true; else echo 'ERROR: app.py not found (expected /app/app.py or /app/src/app.py)'; ls -la /app; ls -la /app/src || true; exit 1; fi"]