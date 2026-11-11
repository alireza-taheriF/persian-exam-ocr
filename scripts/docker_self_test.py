#!/usr/bin/env python3
"""Pre-flight check run by the container before the web server starts.

Fails (exit 1) if anything the app needs at runtime is missing, so a broken
deployment is visible immediately instead of on the first upload.
Set SELFTEST_OCR=1 to also load the recognizer and read one line (slower).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_MODELS = [
    os.environ.get("QWEN_MODEL", "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"),
    "vikp/surya_det3",
    "vikp/surya_layout3",
    "vikp/surya_order",
]


def check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  [ok]   {label}{': ' + detail if detail else ''}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}: {exc}")
        return False


def models_offline() -> str:
    from huggingface_hub import snapshot_download
    for repo in REQUIRED_MODELS:
        snapshot_download(repo, local_files_only=True)
    return f"{len(REQUIRED_MODELS)} models found in {os.environ.get('HF_HOME', '~/.cache/huggingface')}"


def compute() -> str:
    import torch
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU" + (" (MPS available)" if torch.backends.mps.is_available() else "")


def ruled_lines() -> str:
    import cv2
    from pipeline.preprocess import enhance_page_rgb
    from pipeline.ruled import find_ruled_lines
    img = cv2.imread(str(ROOT / "docs" / "images" / "test_sheet_ewc.jpg"))
    assert img is not None, "reference image missing"
    boxes = find_ruled_lines(enhance_page_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) or []
    assert len(boxes) >= 6, f"only {len(boxes)} ruled lines found"
    return f"{len(boxes)} lines on the reference sheet"


def word_export() -> str:
    import tempfile
    from pipeline.export import export_to_word
    out = Path(tempfile.gettempdir()) / "selftest.docx"
    export_to_word([{"index": 1, "type": "text", "text": "آزمون سامانه", "crop_image": None, "illegible": False}], str(out))
    assert out.stat().st_size > 1000
    return ""


def lexicon() -> str:
    from pipeline.corrector import get_lexicon
    n = len(get_lexicon())
    assert n >= 50000, f"lexicon too small ({n})"
    return f"{n:,} words"


def one_line_ocr() -> str:
    from PIL import Image
    from htr_eval.engines import load_engine
    img = Image.open(ROOT / "test_fixtures" / "sample_exam.png").convert("RGB")
    text = load_engine(os.environ.get("OCR_ENGINE", "qwen3vl-fa")).recognize(img.crop((0, 0, img.width, 120)))["text"]
    return repr(text[:40])


def main() -> int:
    print("Pre-flight self-test")
    results = [
        check("model weights available offline", models_offline),
        check("compute device", compute),
        check("ruled-notebook line finder", ruled_lines),
        check("Word (.docx) export", word_export),
        check("Persian lexicon", lexicon),
    ]
    if os.environ.get("SELFTEST_OCR") == "1":
        results.append(check("line recognizer", one_line_ocr))
    if not all(results):
        print("Self-test FAILED")
        return 1
    print("Self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
