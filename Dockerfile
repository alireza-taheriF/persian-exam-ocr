# Persian Exam OCR: self-contained, fully offline image.
#
# 1) ./scripts/prepare_models.sh          (puts the model weights in models/hf)
# 2) docker buildx build --platform linux/amd64 -t persian-exam-ocr:1.0 .
#
# GPU variant (NVIDIA): add  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
# Mirrors (e.g. inside Iran): --build-arg BASE_IMAGE=docker.arvancloud.ir/library/python:3.12-slim
#                             --build-arg PIP_INDEX_URL=https://mirror-pypi.runflare.com/simple
#                             --build-arg APT_MIRROR=http://repo.iut.ac.ir/debian
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.12.0
# Optional Debian mirror, e.g. http://repo.iut.ac.ir/debian
ARG APT_MIRROR=
ARG TORCHVISION_VERSION=0.27.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN if [ -n "${APT_MIRROR}" ]; then \
      printf 'Types: deb\nURIs: %s\nSuites: trixie trixie-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' "${APT_MIRROR}" \
        > /etc/apt/sources.list.d/debian.sources; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

# torch is installed once in the system interpreter and shared by both environments.
# If wheels/ holds torch/torchvision .whl files they are used (handy when the torch index is
# unreachable from the build network); otherwise they come from TORCH_INDEX.
# Dependencies come from PIP_INDEX_URL, so a PyPI mirror is enough besides the torch index.
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    pip install --index-url "${PIP_INDEX_URL}" filelock typing-extensions sympy networkx jinja2 fsspec setuptools \
      numpy==2.4.6 pillow==10.4.0 \
 && if ls /tmp/wheels/torch-*.whl >/dev/null 2>&1; then \
      pip install --no-deps /tmp/wheels/*.whl; \
    elif echo "${TORCH_INDEX}" | grep -q '/cpu'; then \
      pip install --no-deps --index-url "${TORCH_INDEX}" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}; \
    else \
      pip install --index-url "${TORCH_INDEX}" --extra-index-url "${PIP_INDEX_URL}" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}; \
    fi \
 && python -c "import torch, torchvision; print('torch', torch.__version__)"

WORKDIR /app
COPY requirements-app.txt requirements-ocr.txt ./
# Web app + Surya layout (transformers 4.45)
RUN python -m venv --system-site-packages /opt/app \
 && /opt/app/bin/pip install --index-url "${PIP_INDEX_URL}" -r requirements-app.txt
# Line recognizer worker (transformers 5.x)
RUN python -m venv --system-site-packages /opt/ocr \
 && /opt/ocr/bin/pip install --index-url "${PIP_INDEX_URL}" -r requirements-ocr.txt

# Model weights first (large, rarely change), then the source. Nothing is downloaded at runtime.
COPY models/hf /app/models/hf
COPY main.py LICENSE README.md /app/
COPY pipeline /app/pipeline
COPY htr_eval /app/htr_eval
COPY training /app/training
COPY scripts /app/scripts
COPY static /app/static
COPY docs /app/docs
COPY test_fixtures /app/test_fixtures

ENV PATH=/opt/app/bin:$PATH \
    HF_HOME=/app/models/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OCR_ENGINE=qwen3vl-fa \
    OCR_PYTHON=/opt/ocr/bin/python \
    OCR_ALLOW_REMOTE=0 \
    OCR_POST_CORRECT=0 \
    TOKENIZERS_PARALLELISM=false

RUN mkdir -p uploads outputs logs \
 && useradd --create-home --uid 1000 ocr \
 && chown -R ocr:ocr /app/uploads /app/outputs /app/logs
USER ocr

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/bin/bash", "/app/scripts/docker_entrypoint.sh"]
