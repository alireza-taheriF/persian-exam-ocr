#!/usr/bin/env bash
# One-command install on a Linux server with Docker (no large files to transfer).
# Downloads the models from Hugging Face, builds the offline image, and starts the service.
#
#   git clone https://github.com/alireza-taheriF/persian-exam-ocr.git && cd persian-exam-ocr
#   ./scripts/install.sh                 # international mirrors
#   IRAN_MIRRORS=1 ./scripts/install.sh  # Docker/PyPI/Debian mirrors inside Iran
#
# Optional: GPU=1 builds the NVIDIA (CUDA) variant; PORT=8000 changes the published port;
# PLATFORM=linux/amd64 forces the target architecture (default: the server's own).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${IRAN_MIRRORS:-0}" == "1" ]]; then
  BASE_IMAGE="${BASE_IMAGE:-docker.arvancloud.ir/library/python:3.12-slim}"
  PIP_INDEX_URL="${PIP_INDEX_URL:-https://package-mirror.liara.ir/repository/pypi/simple}"
  APT_MIRROR="${APT_MIRROR:-https://mirror.iranserver.com/debian}"
  PIP_FALLBACK_INDEX="${PIP_FALLBACK_INDEX:-https://mirror-pypi.runflare.com/simple}"
fi
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
APT_MIRROR="${APT_MIRROR:-}"
PIP_FALLBACK_INDEX="${PIP_FALLBACK_INDEX:-https://pypi.org/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
[[ "${GPU:-0}" == "1" ]] && TORCH_INDEX="https://download.pytorch.org/whl/cu126"
TAG="persian-exam-ocr:1.0"

command -v docker >/dev/null || { echo "Docker is required: https://docs.docker.com/engine/install/"; exit 1; }

echo "==> [1/3] Model weights (~4.8 GB, one-time download from Hugging Face)"
MODELS="mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0 vikp/surya_det3 vikp/surya_layout3 vikp/surya_order"
mkdir -p models/hf
for attempt in 1 2 3 4 5; do
  if docker run --rm -i -v "$PWD/models/hf:/hf" -e HF_HOME=/hf -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    "$BASE_IMAGE" sh -c "pip install -q --retries 10 --index-url '$PIP_INDEX_URL' huggingface_hub==0.36.2 && python - $MODELS" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print("   ", repo, flush=True)
    snapshot_download(repo)  # resumes if interrupted
PY
  then break; fi
  [[ $attempt == 5 ]] && { echo "Model download failed 5 times; check the connection to Hugging Face (or set HF_ENDPOINT to a mirror) and re-run."; exit 1; }
  echo "    network error, retrying in 20 s (attempt $attempt/5)"; sleep 20
done
# Make the cache plain files owned by the invoking user (Docker wrote it as root).
docker run --rm -v "$PWD/models/hf:/hf" "$BASE_IMAGE" sh -c "chmod -R a+rX /hf" || true

echo "==> [2/3] Building the offline image ($TAG)"
# Download every Python package into wheels/ first. pip keeps finished files, so a dropped
# connection only costs the file in flight; re-running resumes. The image build then installs
# from wheels/ with no network. (GPU builds skip this and install from the indexes.)
if [[ "${GPU:-0}" != "1" ]]; then
  if [[ "${PLATFORM:-}" == "linux/amd64" || ( -z "${PLATFORM:-}" && "$(uname -m)" == "x86_64" ) ]]; then
    # CPU torch from this repo's GitHub release (github.com is reachable where download.pytorch.org may not be)
    REL="https://github.com/alireza-taheriF/persian-exam-ocr/releases/download/v1.0"
    for w in torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl torchvision-0.27.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl; do
      [[ -s "wheels/$w" ]] && continue
      echo "    $w"
      curl -fL --retry 10 --retry-all-errors -C - -o "wheels/$w.part" "$REL/${w//+/%2B}" && mv "wheels/$w.part" "wheels/$w"
    done
  fi
  echo "    Python packages -> wheels/"
  for attempt in 1 2 3 4 5 6; do
    IDX="$PIP_INDEX_URL"; (( attempt % 2 == 0 )) && IDX="$PIP_FALLBACK_INDEX"
    if docker run --rm ${PLATFORM:+--platform "$PLATFORM"} -v "$PWD/wheels:/w" -v "$PWD:/src:ro" "$BASE_IMAGE" sh -c "
        set -e; D='pip download -q --dest /w --find-links /w --index-url $IDX --retries 10 --timeout 120 --only-binary=:all:'
        \$D torch==2.12.0 torchvision==0.27.0
        \$D -r /src/requirements-app.txt
        \$D -r /src/requirements-ocr.txt"; then
      touch wheels/.complete; break
    fi
    [[ $attempt == 6 ]] && { echo "Package download keeps failing; re-run the script later to resume."; exit 1; }
    echo "    download interrupted, resuming in 20 s (attempt $attempt/6)"; sleep 20
  done
fi

for attempt in 1 2 3; do   # finished layers are cached, so a retry only redoes the failed step
  if DOCKER_BUILDKIT=1 docker build -t "$TAG" ${PLATFORM:+--platform "$PLATFORM"} \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" \
    --build-arg APT_MIRROR="$APT_MIRROR" --build-arg TORCH_INDEX="$TORCH_INDEX" \
    --build-arg PIP_FALLBACK_INDEX="$PIP_FALLBACK_INDEX" .; then break; fi
  [[ $attempt == 3 ]] && { echo "Image build failed 3 times (network?). Re-run the script to continue."; exit 1; }
  echo "    build step failed, retrying in 30 s (attempt $attempt/3)"; sleep 30
done

echo "==> [3/3] Starting the service"
mkdir -p outputs logs
docker run --rm -v "$PWD:/w" "$BASE_IMAGE" chown 1000:1000 /w/outputs /w/logs || chmod 777 outputs logs
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
