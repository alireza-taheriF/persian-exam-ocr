"""
Text extraction for exam blocks.

Layout detection uses Surya. Each text/list block is split into lines (Surya line
detection) and every line is read by a LOCAL engine (htr_eval/engines.py, run by
htr_eval/ocr_worker.py in its own interpreter). Nothing leaves this machine.

The old remote path (OpenAI-compatible vision API) only runs when OCR_ALLOW_REMOTE=1;
then block crops are uploaded to VLM_API_BASE.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import RateLimitError
from pipeline.openai_compat import (
    create_openai_client,
    resolve_openai_api_key,
    resolve_openai_base_url,
)
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline.crop import crop_block, should_crop

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

_OCR_BLOCK_TYPES = frozenset({"text", "list"})
_OCR_MIN_TEXT_LENGTH = 2
_OCR_MIN_CROP_HEIGHT = int(os.environ.get("OCR_MIN_CROP_HEIGHT", "64"))
_OCR_UPSCALE_FACTOR = float(os.environ.get("OCR_UPSCALE_FACTOR", "2.0"))
_OCR_REQUEST_DELAY_SEC = float(os.environ.get("OCR_REQUEST_DELAY_SEC", "2.5"))
_PLACEHOLDER_API_KEYS = frozenset({"", "your_api_key_here", "changeme", "replace_me"})

_SYSTEM_PROMPT = (
    "تو یک موتور OCR بسیار دقیق برای زبان فارسی هستی. "
    "تصویر بخشی از یک برگه امتحان دانشگاهی است و ممکن است چاپی یا دست‌نویس باشد. "
    "متن را دقیقاً همان‌طور که در تصویر نوشته شده استخراج کن؛ "
    "از اصلاح املا، تکمیل جمله، ترجمه یا توضیح اضافه خودداری کن. "
    "اعداد، علائم نگارشی، پرانتز و نقطه‌گذاری را حفظ کن. "
    "اگر بخشی کاملاً ناخوانا است فقط همان بخش را با [ناخوانا] علامت بزن. "
    "فقط متن استخراج‌شده را برگردان، بدون مقدمه یا جمع‌بندی."
)

_USER_PROMPT = (
    "متن فارسی داخل این تصویر (چاپی یا دست‌نویس) را خط‌به‌خط و دقیق استخراج کن."
)

# A short unit repeated back-to-back for 24+ chars is a decoding loop, never handwriting.
_LOOP_RE = re.compile(r"(.{1,12}?)\1{2,}")

_WORKER: subprocess.Popen | None = None
_WORKER_LOCK = threading.Lock()
_LINE_PAD = 8

_VLM_CLIENT: object | None = None
_VLM_CONFIG_ERROR: str | None = None


def _get_api_key() -> str:
    return resolve_openai_api_key(dict(os.environ))


def _remote_enabled() -> bool:
    return os.environ.get("OCR_ALLOW_REMOTE", "").strip() == "1"


def _validate_vlm_config() -> str | None:
    """Return an error message when the remote VLM API is off or not configured."""
    if not _remote_enabled():
        return "Remote VLM is off: set OCR_ALLOW_REMOTE=1 to opt in (images then leave this machine)."
    api_key = _get_api_key()
    if not api_key or api_key.lower() in _PLACEHOLDER_API_KEYS:
        return (
            "VLM_API_KEY is missing or still set to the placeholder value. "
            "Copy .env.example to .env and set a valid API key."
        )
    return None


def _get_vlm_client():
    global _VLM_CLIENT, _VLM_CONFIG_ERROR

    config_error = _validate_vlm_config()
    if config_error:
        _VLM_CONFIG_ERROR = config_error
        raise ValueError(config_error)

    if _VLM_CLIENT is not None:
        return _VLM_CLIENT

    api_base = resolve_openai_base_url(dict(os.environ))
    _VLM_CLIENT = create_openai_client(api_key=_get_api_key(), base_url=api_base)
    _VLM_CONFIG_ERROR = None
    return _VLM_CLIENT


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Convert grayscale, BGR, or RGB numpy image to PIL RGB."""
    if len(image.shape) == 2:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(image)


def _crop_region(pil_image: Image.Image, bbox: list) -> Image.Image:
    x1, y1, x2, y2 = bbox
    img_w, img_h = pil_image.size
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop bbox: {bbox}")
    return pil_image.crop((x1, y1, x2, y2))


def _pil_to_base64(image: Image.Image, *, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _clean_vlm_output(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:\w*)\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _message_content_to_text(content: object) -> str:
    """Normalize OpenAI-compatible message content (str or multimodal parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _clean_vlm_output(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text = part.get("text")
                    if text:
                        parts.append(str(text))
            elif isinstance(part, str):
                parts.append(part)
        return _clean_vlm_output("\n".join(parts))
    return _clean_vlm_output(str(content))


def _is_illegible_text(text: str) -> bool:
    stripped = text.strip()
    return not stripped or len(stripped) < _OCR_MIN_TEXT_LENGTH


def _mark_illegible(block: dict, original_image: np.ndarray, *, reason: str | None = None) -> None:
    block["illegible"] = True
    block["text"] = None
    block["crop_image"] = crop_block(original_image, block["bbox"], rgb_input=True)
    if reason:
        block["vlm_error"] = reason


def _prepare_ocr_region(region: Image.Image) -> Image.Image:
    """Upscale small handwriting regions so the VLM can read fine strokes."""
    width, height = region.size
    if height >= _OCR_MIN_CROP_HEIGHT and width >= _OCR_MIN_CROP_HEIGHT:
        return region

    scale = max(
        _OCR_UPSCALE_FACTOR,
        _OCR_MIN_CROP_HEIGHT / max(height, 1),
        _OCR_MIN_CROP_HEIGHT / max(width, 1),
    )
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return region.resize(new_size, Image.Resampling.LANCZOS)


def extract_text_with_vlm(base64_image: str) -> str:
    """
    Send a base64-encoded image to the configured OpenAI-compatible VLM (opt-in), else local Ollama vision.
    """
    # Next try OpenAI-compatible VLM
    try:
        client = _get_vlm_client()
        model = os.environ.get("VLM_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ],
            timeout=30,
        )
        if response.choices and response.choices[0].message:
            txt = _message_content_to_text(response.choices[0].message.content)
            if txt:
                return txt
    except Exception as exc:
        logger.warning("OpenAI-compatible VLM failed: %s", exc)

    # Fallback to local Ollama Vision
    try:
        import requests
        base = os.environ.get("CORRECTOR_BASE_URL", "http://localhost:11434").rstrip("/")
        body = {
            "model": "qwen3-vl:8b-instruct",
            "messages": [{"role": "user", "content": _SYSTEM_PROMPT + "\n" + _USER_PROMPT, "images": [base64_image]}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0}
        }
        r = requests.post(f"{base}/api/chat", json=body, timeout=45)
        if r.status_code == 200:
            return r.json()["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Local Ollama vision failed: %s", exc)

    return ""


def _python_for(engine: str | None) -> str:
    """Interpreter for an engine: kraken lives in its own venv (OCR_KRAKEN_PYTHON), the rest in OCR_PYTHON."""
    root = _PROJECT_ROOT
    if engine and engine.startswith("kraken"):
        return os.environ.get("OCR_KRAKEN_PYTHON") or str(root / ".venv-kraken" / "bin" / "python")
    venv_python = root / ".venv" / "bin" / "python"
    return os.environ.get("OCR_PYTHON") or (str(venv_python) if venv_python.exists() else sys.executable)


def _start_worker(engine: str | None = None) -> subprocess.Popen:
    python = _python_for(engine)
    env = {**os.environ, "OCR_ENGINE": engine} if engine else None
    proc = subprocess.Popen([python, "-m", "htr_eval.ocr_worker"], cwd=_PROJECT_ROOT, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env)
    if '"ready"' not in (proc.stdout.readline() or ""):
        raise RuntimeError(f"local OCR worker failed to start with {python} (see its log above)")
    return proc


def _worker() -> subprocess.Popen:
    """Start (once) the local OCR worker; OCR_PYTHON must have torch + transformers>=5."""
    global _WORKER
    if _WORKER is None or _WORKER.poll() is not None:
        _WORKER = _start_worker()
    return _WORKER


def recognize_line(line: Image.Image) -> str:
    """Read one text-line image with the local engine (OCR_ENGINE, default qwen3vl-fa)."""
    with _WORKER_LOCK:
        w = _worker()
        w.stdin.write(json.dumps({"png": _pil_to_base64(line)}) + "\n")
        w.stdin.flush()
        reply = w.stdout.readline()
    if not reply:
        raise RuntimeError("local OCR worker died")
    return json.loads(reply)["text"]


_COUNTING_RE = re.compile(r"۱۲۳۴۵۶|۲۳۴۵۶۷|123456|234567")
_PROMPT_ECHO = ("رونویسی کن", "داخل این تصویر", "رشته خالی", "رشتهٔ خالی")  # a model that answers with (part of) our own instruction


def _is_loop(text: str) -> bool:
    """Decoding junk: a repeated unit, a 1-2-3-4-5-6 counting run, or the model repeating the prompt."""
    return (any(m.end() - m.start() >= 24 for m in _LOOP_RE.finditer(text)) or bool(_COUNTING_RE.search(text))
            or any(p in text for p in _PROMPT_ECHO))


def _fit_line_height(crop: Image.Image, min_h: int = 64, target_h: int = 96, max_scale: float = 6.0) -> Image.Image:
    """Chat/phone photos give 10-20 px lines; the engines were measured on ~100 px crops, so scale small ones up."""
    if crop.height >= min_h:
        return crop
    k = min(max_scale, target_h / max(1, crop.height))
    return crop.resize((round(crop.width * k), round(crop.height * k)), Image.Resampling.LANCZOS)


def _line_boxes(region: Image.Image) -> list[list[float]]:
    """Surya text-line boxes inside a region, top to bottom."""
    from surya.detection import batch_text_detection

    from pipeline.layout import det_model, det_processor

    return sorted((b.bbox for b in batch_text_detection([region], det_model, det_processor)[0].bboxes),
                  key=lambda b: b[1] + b[3])


def _looks_like_text(region: Image.Image) -> bool:
    """Surya's layout calls big handwritten paragraphs "figure". A real figure has few wide line-shaped boxes."""
    return sum(1 for x1, y1, x2, y2 in _line_boxes(region) if (x2 - x1) >= 3 * max(1, y2 - y1)) >= 3


def _split_lines(region: Image.Image) -> list[Image.Image]:
    """Surya line detection inside a block, top to bottom (the whole block if none found)."""
    if region.height <= 85:
        return [_fit_line_height(region)]
    boxes = _line_boxes(region)
    w, h = region.size
    crops = [region.crop((max(0, int(x1) - _LINE_PAD), max(0, int(y1) - _LINE_PAD),
                          min(w, int(x2) + _LINE_PAD), min(h, int(y2) + _LINE_PAD))) for x1, y1, x2, y2 in boxes]
    return [_fit_line_height(c) for c in (crops or [region])]


_ENSEMBLE_DEFAULT = "qwen3vl-fa,ollama-vision:qwen3-vl:8b-instruct"


def _rescue_figures(blocks: list[dict], pil_image: Image.Image) -> None:
    """Handwriting that layout mislabelled as a big "figure" would otherwise never be read: relabel it as text."""
    for blk in blocks:
        if blk.get("type") == "figure" and should_crop(blk):
            x1, y1, x2, y2 = blk["bbox"]
            if (x2 - x1) > 200 and (y2 - y1) > 100:
                try:
                    if _looks_like_text(_crop_region(pil_image, blk["bbox"])):
                        blk["type"] = "text"
                except Exception as exc:
                    logger.warning("figure check failed for %s: %s", blk["bbox"], exc)


def _ensemble_engines() -> list[str]:
    """OCR_ENSEMBLE=1 -> the measured default set; =a,b,c -> that list; unset/0 -> single engine (OCR_ENGINE)."""
    v = os.environ.get("OCR_ENSEMBLE", "").strip()
    if v in ("", "0"):
        return []
    return _ENSEMBLE_DEFAULT.split(",") if v == "1" else [e.strip() for e in v.split(",") if e.strip()]


def _read_all(engine: str, crops: list[Image.Image]) -> list[str]:
    """One engine over every crop, then shut it down: only one big model sits in memory at a time."""
    proc = _start_worker(engine)
    try:
        out = []
        for crop in crops:
            proc.stdin.write(json.dumps({"png": _pil_to_base64(crop)}) + "\n")
            proc.stdin.flush()
            reply = proc.stdout.readline()
            if not reply:
                raise RuntimeError(f"engine {engine} died")
            out.append(json.loads(reply)["text"])
        return out
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if engine.startswith("ollama-vision:"):  # free Ollama's copy before the next engine loads (localhost only)
            import requests
            base = os.environ.get("CORRECTOR_BASE_URL", "http://localhost:11434").rstrip("/")
            model = engine[len("ollama-vision:"):].partition("@")[0]
            try:
                requests.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
            except requests.RequestException:
                pass


def _fuse_lines(per_engine: list[list[str]]) -> list[str]:
    """Per line: drop empty/looping candidates, then word-vote (htr_eval.ensemble). Empty if nothing usable."""
    from htr_eval.ensemble import fuse
    from pipeline.normalizer import normalize_text
    fused = []
    for cands in zip(*per_engine):
        usable = [normalize_text(c) for c in cands if c.strip() and not _is_loop(c)]
        fused.append(fuse(usable) if usable else "")
    return fused


def _ensemble_lines(crops: list[Image.Image], engines: list[str]) -> list[str]:
    runs = []
    for e in engines:
        try:
            runs.append(_read_all(e, crops))
        except Exception as exc:  # one broken engine must not sink the page
            logger.warning("ensemble engine %s skipped: %s", e, exc)
    if not runs:
        raise RuntimeError("every ensemble engine failed")
    return _fuse_lines(runs)


def get_ocr_status() -> dict[str, str | bool]:
    """Return OCR engine configuration for health checks (no secrets)."""
    if not _remote_enabled():
        return {"engine": "local", "configured": True, "model": os.environ.get("OCR_ENGINE", "qwen3vl-fa"),
                "api_base": "local", "message": "ready"}
    config_error = _validate_vlm_config()
    return {
        "engine": "vlm",
        "configured": config_error is None,
        "model": os.environ.get("VLM_MODEL", "gpt-4o-mini"),
        "api_base": os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"),
        "message": config_error or "ready",
    }


def _wants_ocr(block: dict) -> bool:
    """True when this block should be sent to the VLM (not embedded as image only)."""
    if should_crop(block):
        return False
    block_type = block.get("type")
    if block_type in _OCR_BLOCK_TYPES:
        return True
    # Surya often labels small handwriting crops as Figure/Picture.
    return block_type == "figure"


def run_ocr(original_image: np.ndarray, blocks: list[dict]) -> list[dict]:
    """
    For each block:
    - table/figure: crop directly (no VLM call)
    - text/list: split into lines → local engine per line (or the remote VLM per block
      when OCR_ALLOW_REMOTE=1); fallback to crop on failure
    """
    pil_image = _to_pil_rgb(original_image)
    remote = _remote_enabled()
    config_error = _validate_vlm_config() if remote else None
    if config_error:
        logger.warning("%s Text blocks will be exported as crops.", config_error)

    if not remote and not config_error:
        _rescue_figures(blocks, pil_image)

    ensemble = [] if remote else _ensemble_engines()
    fused: dict[int, list[str]] = {}
    if ensemble and not config_error:
        pending = [(i, [_fit_line_height(_crop_region(pil_image, b["bbox"]))] if b.get("is_line") else _split_lines(_crop_region(pil_image, b["bbox"]))) for i, b in enumerate(blocks)
                   if not should_crop(b) and _wants_ocr(b)]
        flat = [c for _, crops in pending for c in crops]
        texts = _ensemble_lines(flat, ensemble) if flat else []
        pos = 0
        for i, crops in pending:
            fused[i] = texts[pos:pos + len(crops)]
            pos += len(crops)

    ocr_calls = 0
    for index, block in enumerate(blocks):
        if should_crop(block):
            block["text"] = None
            block["illegible"] = False
            block["crop_image"] = crop_block(original_image, block["bbox"], rgb_input=True)
            continue

        block_type = block.get("type")
        if not _wants_ocr(block):
            block["text"] = None
            block["crop_image"] = None
            block["illegible"] = False
            continue

        if config_error:
            _mark_illegible(block, original_image, reason=config_error)
            continue

        try:
            region = _crop_region(pil_image, block["bbox"])
            if remote:
                if ocr_calls > 0 and _OCR_REQUEST_DELAY_SEC > 0:
                    time.sleep(_OCR_REQUEST_DELAY_SEC)
                text = extract_text_with_vlm(_pil_to_base64(_prepare_ocr_region(region)))
                ocr_calls += 1
            elif ensemble:
                text = "\n".join(t for t in fused[index] if t.strip())
            else:
                line_crops = [_fit_line_height(region)] if block.get("is_line") else _split_lines(region)
                lines = map(recognize_line, line_crops)
                text = "\n".join(t for t in lines if t.strip() and not _is_loop(t))

            if text and os.environ.get("OCR_POST_CORRECT", "1") == "1":
                try:
                    from pipeline.corrector import correct_lines
                    block_lines = [l for l in text.split("\n") if l.strip()]
                    if block_lines:
                        text = "\n".join(correct_lines(block_lines))
                except Exception as corr_exc:
                    logger.warning("Post-correction skipped: %s", corr_exc)

            if _is_illegible_text(text):
                logger.warning("VLM returned empty or short text for block %s", block["bbox"])
                _mark_illegible(block, original_image, reason="empty_or_short_vlm_output")
                continue

            block["text"] = text
            block["crop_image"] = None
            block["illegible"] = False

        except Exception as exc:
            logger.warning("VLM OCR failed on block %s: %s", block["bbox"], exc)
            _mark_illegible(block, original_image, reason=str(exc))

    return blocks
