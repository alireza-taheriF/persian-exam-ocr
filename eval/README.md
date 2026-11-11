# eval/

`reports/` holds the outputs of `scripts/eval_release.py`, the same code path the web app serves.

| File | What it measures |
|---|---|
| `qwen3vl-fa_gold240.json` | Default engine on the gold set (240 lines, 4 documents, 168 legible lines scored) |
| `qwen3vl-fa_gold240_corrected.json` | Same predictions + local LLM post-correction (`OCR_POST_CORRECT=1`) |
| `qwen3vl-fa-lora-v2_gold240.json` | The project's LoRA fine-tune (`adapters/qwen3vl-fa-lora-v2`, `OCR_ENGINE=qwen3vl-lora`) |
| `*.preds.jsonl` | Per-line predictions behind each report |
| `ewc_sheet_offline.json` | Full-page run of the app on `docs/images/test_sheet_ewc.jpg` against `ewc_sheet_ground_truth.txt` |

The gold line images are crops of third-party handwritten notes and are not redistributed here;
they are available to reviewers on request. Metrics: corpus-level CER/WER after Persian
normalization (`pipeline/normalizer.py`); character accuracy = 1 − CER, word accuracy = 1 − WER.
