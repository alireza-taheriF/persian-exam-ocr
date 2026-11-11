# adapters/

`qwen3vl-fa-lora-v2/`: LoRA (r=16, α=32) fine-tuned by this project on Persian handwritten
exam-note lines, on top of `mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0`.

Enable it with `OCR_ENGINE=qwen3vl-lora` (in `.env` or `docker-compose.yml`). On the gold set it
scores 63.5% character accuracy against 63.9% for the base model (within the margin of error):
better on one document, worse on others, and it more often writes text for illegible lines.
See `eval/reports/qwen3vl-fa-lora-v2_gold240.json`.
