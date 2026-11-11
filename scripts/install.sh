#!/usr/bin/env bash
# One-command install on a Linux server with Docker (no large files to transfer).
# Downloads the models from Hugging Face, builds the offline image, and starts the service.
#
#   git clone https://github.com/alireza-taheriF/persian-exam-ocr.git && cd persian-exam-ocr
#   ./scripts/install.sh                 # international mirrors
#   IRAN_MIRRORS=1 ./scripts/install.sh  # Docker/PyPI/Debian mirrors inside Iran
#
# Optional: GPU=1 builds the NVIDIA (CUDA) variant; PORT=8000 changes the published port.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${IRAN_MIRRORS:-0}" == "1" ]]; then
  BASE_IMAGE="${BASE_IMAGE:-docker.arvancloud.ir/library/python:3.12-slim}"
  PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror-pypi.runflare.com/simple}"
  APT_MIRROR="${APT_MIRROR:-http://repo.iut.ac.ir/debian}"
fi
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
APT_MIRROR="${APT_MIRROR:-}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
[[ "${GPU:-0}" == "1" ]] && TORCH_INDEX="https://download.pytorch.org/whl/cu126"
TAG="persian-exam-ocr:1.0"

command -v docker >/dev/null || { echo "Docker is required: https://docs.docker.com/engine/install/"; exit 1; }

echo "==> [1/3] Model weights (~4.8 GB, one-time download from Hugging Face)"
MODELS="mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0 vikp/surya_det3 vikp/surya_layout3 vikp/surya_order"
mkdir -p models/hf
docker run --rm -i -v "$PWD/models/hf:/hf" -e HF_HOME=/hf -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  "$BASE_IMAGE" sh -c "pip install -q --index-url '$PIP_INDEX_URL' huggingface_hub==0.36.2 && python - $MODELS" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print("   ", repo, flush=True)
    snapshot_download(repo)  # resumes if interrupted
PY
# Make the cache plain files owned by the invoking user (Docker wrote it as root).
docker run --rm -v "$PWD/models/hf:/hf" "$BASE_IMAGE" sh -c "chmod -R a+rX /hf"

echo "==> [2/3] Building the offline image ($TAG)"
DOCKER_BUILDKIT=1 docker build -t "$TAG" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" \
  --build-arg APT_MIRROR="$APT_MIRROR" --build-arg TORCH_INDEX="$TORCH_INDEX" .

echo "==> [3/3] Starting the service"
mkdir -p outputs logs
docker run --rm -v "$PWD:/w" "$BASE_IMAGE" chown 1000:1000 /w/outputs /w/logs
if docker compose version >/dev/null 2>&1; then
  PORT="${PORT:-8000}" docker compose up -d
else
  docker rm -f persian-exam-ocr >/dev/null 2>&1 || true
  docker run -d --name persian-exam-ocr --restart unless-stopped -p "${PORT:-8000}:8000" \
    -v "$PWD/outputs:/app/outputs" -v "$PWD/logs:/app/logs" "$TAG"
fi

echo -n "    waiting for the self-test and model load"
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${PORT:-8000}/health" >/dev/null 2>&1; then
    echo; echo "Ready: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT:-8000}"; exit 0
  fi
  echo -n "."; sleep 5
done
echo; echo "Service did not become healthy; see: docker logs persian-exam-ocr"; exit 1
