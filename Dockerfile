# Persian Exam OCR: self-contained, fully offline image.
#
# 1) ./scripts/prepare_models.sh          (puts the model weights in models/hf)
# 2) docker buildx build --platform linux/amd64 -t persian-exam-ocr:1.0 .
#
# GPU variant (NVIDIA): add  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
# Mirrors (e.g. inside Iran): --build-arg BASE_IMAGE=docker.arvancloud.ir/library/python:3.12-slim
#                             --build-arg PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple
#                             --build-arg APT_MIRROR=https://mirror.iranserver.com/debian
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.12.0
# Optional Debian mirror, e.g. https://mirror.iranserver.com/debian
ARG APT_MIRROR=
ARG TORCHVISION_VERSION=0.27.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN if [ -n "${APT_MIRROR}" ]; then \
      printf 'Types: deb\nURIs: %s\nSuites: trixie trixie-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' "${APT_MIRROR}" \
        > /etc/apt/sources.list.d/debian.sources; fi \
 && echo "Acquire::Retries \"10\";" > /etc/apt/apt.conf.d/80retries \
 && apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

# All Python packages are installed from wheels/ when scripts/install.sh has filled it
# (wheels/.complete); the build then needs no package index at all. Otherwise they come from
# PIP_INDEX_URL (+ TORCH_INDEX for torch). torch lives in the system interpreter and is shared
# by both environments.
ARG PIP_FALLBACK_INDEX=https://pypi.org/simple
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    if [ -f /tmp/wheels/.complete ]; then \
      pip install --no-index --find-links /tmp/wheels torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}; \
    else \
      pip install --index-url "${PIP_INDEX_URL}" filelock typing-extensions sympy networkx jinja2 fsspec setuptools \
        numpy==2.4.6 pillow==10.4.0 \
      && if echo "${TORCH_INDEX}" | grep -q '/cpu'; then \
        pip install --no-deps --index-url "${TORCH_INDEX}" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        || pip install --index-url "${PIP_INDEX_URL}" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}; \
      else \
        pip install --index-url "${TORCH_INDEX}" --extra-index-url "${PIP_INDEX_URL}" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}; \
      fi; \
    fi \
 && python -c "import torch, torchvision; print('torch', torch.__version__)"

WORKDIR /app
COPY requirements-app.txt requirements-ocr.txt ./
# Web app + Surya layout (transformers 4.45) and the line-recognizer worker (transformers 5.x)
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    python -m venv --system-site-packages /opt/app \
 && python -m venv --system-site-packages /opt/ocr \
 && if [ -f /tmp/wheels/.complete ]; then \
      SRC="--no-index --find-links /tmp/wheels"; \
      /opt/app/bin/pip install $SRC -r requirements-app.txt \
      && /opt/ocr/bin/pip install $SRC -r requirements-ocr.txt; \
    else \
      { /opt/app/bin/pip install --index-url "${PIP_INDEX_URL}" -r requirements-app.txt \
        || /opt/app/bin/pip install --index-url "${PIP_FALLBACK_INDEX}" -r requirements-app.txt; } \
      && { /opt/ocr/bin/pip install --index-url "${PIP_INDEX_URL}" -r requirements-ocr.txt \
        || /opt/ocr/bin/pip install --index-url "${PIP_FALLBACK_INDEX}" -r requirements-ocr.txt; }; \
    fi

# Model weights first (large, rarely change), then the source. Nothing is downloaded at runtime.
COPY models/hf /app/models/hf
COPY main.py LICENSE README.md /app/
COPY pipeline /app/pipeline
COPY htr_eval /app/htr_eval
COPY adapters /app/adapters
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
