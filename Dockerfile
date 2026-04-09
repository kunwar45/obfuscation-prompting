FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && grep -v '^torch' requirements.txt > requirements-docker.txt \
    && pip install -r requirements-docker.txt

COPY main.py .
COPY make_dataset.py .
COPY run_blackbox.py .
COPY run_framing.py .
COPY run_interp.py .
COPY run_patching.py .
COPY README.md .
COPY src ./src
COPY scripts ./scripts
COPY vertex_jobs ./vertex_jobs
COPY docs ./docs
COPY tests ./tests

RUN mkdir -p /app/data /app/results /app/activations

CMD ["python", "main.py", "--help"]
