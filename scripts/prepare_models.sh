#!/usr/bin/env bash
# Collect the model weights the app needs into ./models/hf (a Hugging Face cache).
# Uses an existing local cache when it has the files, otherwise downloads them.
#   ./scripts/prepare_models.sh                 # downloads (needs internet)
#   SRC_HF_HOME=~/.cache/huggingface ./scripts/prepare_models.sh   # copy from a local cache
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="models/hf"
MODELS=(
  "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
  "vikp/surya_det3"
  "vikp/surya_layout3"
  "vikp/surya_order"
)
mkdir -p "$DEST/hub"
for repo in "${MODELS[@]}"; do
  dir="models--${repo//\//--}"
  if [[ -n "${SRC_HF_HOME:-}" && -d "$SRC_HF_HOME/hub/$dir" ]]; then
    echo "copy  $repo  (from $SRC_HF_HOME)"
    # Dereference symlinks and keep only refs/ + snapshots/ so the result is plain files
    # (portable across machines, safe for Docker COPY).
    mkdir -p "$DEST/hub/$dir"
    rsync -aL --delete --exclude blobs --exclude ".no_exist" "$SRC_HF_HOME/hub/$dir/" "$DEST/hub/$dir/"
  else
    echo "fetch $repo"
    HF_HOME="$PWD/$DEST" python -c "from huggingface_hub import snapshot_download as s; s('$repo')"
  fi
done
HF_HOME="$PWD/$DEST" HF_HUB_OFFLINE=1 python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0", "vikp/surya_det3", "vikp/surya_layout3", "vikp/surya_order"]:
    snapshot_download(r, local_files_only=True)
print("all models present offline")
PY
du -sh "$DEST"
