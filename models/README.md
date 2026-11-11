# models/

Model weights are not stored in git. `scripts/prepare_models.sh` fills this folder:

```
models/hf/            Hugging Face cache (HF_HOME) used fully offline
  hub/models--mohajesmaeili--Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0   line recognizer (~4.3 GB)
  hub/models--vikp--surya_det3                                     line detection
  hub/models--vikp--surya_layout3                                  page layout
  hub/models--vikp--surya_order                                    reading order
```

The Docker image copies `models/hf` to `/app/models/hf` and runs with `HF_HUB_OFFLINE=1`.
