"""One interface for every line-level HTR engine: PIL line crop → {"text", "candidates"}.

All engines run locally. Add a new one by writing a class with ``name`` and
``recognize`` and registering it in ENGINES.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from html import unescape
from typing import Protocol

from PIL import Image


class Engine(Protocol):
    name: str

    def recognize(self, line: Image.Image) -> dict:  # {"text": str, "candidates": list[list[str]] | None}
        ...


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", "1000000"))
QWEN_FA = os.environ.get("QWEN_MODEL", "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0")


class SuryaRec:
    """Baseline: Surya recognition (the project's pinned surya-ocr 0.6.x), Persian lang hint."""
    name = "surya"

    def __init__(self) -> None:
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        from surya.ocr import run_recognition
        self._run, self._model, self._proc = run_recognition, load_model(), load_processor()

    def recognize(self, line: Image.Image) -> dict:
        w, h = line.size
        out = self._run([line], [["fa"]], self._model, self._proc, bboxes=[[[0, 0, w, h]]])
        return {"text": " ".join(t.text for t in out[0].text_lines), "candidates": None}


class HFVisionLM:
    """Any Hugging Face image-text-to-text OCR model (Qwen-VL fine-tunes etc.), greedy decoding."""
    PROMPT = "متن فارسی داخل این تصویر را عیناً رونویسی کن."

    def __init__(self, name: str, model_id: str, max_new_tokens: int = 96, prompt: str | None = None,
                 html: bool = False, adapter: str | None = None, dtype=None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.name, self._max, self._html = name, max_new_tokens, html
        self.PROMPT = prompt or self.PROMPT
        self._dev = _device()
        dtype = dtype or (torch.float32 if self._dev == "cpu" else torch.float16)
        self._proc = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self._dev).eval()
        if adapter:  # LoRA adapter from htr_eval/lora_train.py
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, adapter).eval()

    def recognize(self, line: Image.Image) -> dict:
        import torch
        w, h = line.size
        if w < 32 or h < 32:
            new_w = max(w, 32)
            new_h = max(h, 32)
            padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
            padded.paste(line, ((new_w - w) // 2, (new_h - h) // 2))
            line = padded
        # Very large crops overflow the processor's token budget ("Mismatch in image token count");
        # keep every crop under ~1 MP (about 1,300 vision tokens).
        if line.width * line.height > MAX_PIXELS:
            k = (MAX_PIXELS / (line.width * line.height)) ** 0.5
            line = line.resize((max(32, int(line.width * k)), max(32, int(line.height * k))), Image.Resampling.LANCZOS)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}]
        prompt = self._proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self._proc(images=[line], text=[prompt], return_tensors="pt").to(self._dev)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self._max, do_sample=False,
                                       repetition_penalty=1.05)
        text = self._proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if self._html:  # surya-ocr-2 answers in HTML
            text = " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())
        return {"text": text.strip(), "candidates": None}


class OllamaVision:
    """Any vision model served by local Ollama. Name: ``ollama-vision:<model>[@pad,up,gray]``."""
    PROMPT = ("متن دست‌نویس فارسی داخل این تصویر را دقیقاً همان‌طور که نوشته شده رونویسی کن؛ املا و ارقام نویسنده را حفظ کن. "
              "فقط خود متن را بنویس. اگر خوانا نیست رشتهٔ خالی برگردان.")

    def __init__(self, name: str, model: str, prep: tuple[str, ...] = ()) -> None:
        from urllib.parse import urlparse
        self.name, self._model, self._prep = name, model, prep
        self._base = os.environ.get("CORRECTOR_BASE_URL", "http://localhost:11434").rstrip("/")
        if urlparse(self._base).hostname not in {"localhost", "127.0.0.1"}:
            raise SystemExit(f"refusing non-local Ollama host {self._base!r}: nothing may leave this machine")

    def _preprocess(self, im: Image.Image) -> Image.Image:
        from PIL import ImageOps
        if "gray" in self._prep:
            im = ImageOps.autocontrast(im.convert("L"), cutoff=1).convert("RGB")
        if "up" in self._prep and im.height < 96:
            im = im.resize((round(im.width * 96 / im.height), 96), Image.Resampling.LANCZOS)
        if "pad" in self._prep:
            im = ImageOps.expand(im, 8, fill="white")
        return im

    def recognize(self, line: Image.Image) -> dict:
        import base64
        from io import BytesIO

        import requests
        buf = BytesIO()
        self._preprocess(line).save(buf, format="PNG")
        body = {"model": self._model, "stream": False, "options": {"temperature": 0, "num_predict": 160},
                "messages": [{"role": "user", "content": self.PROMPT,
                              "images": [base64.b64encode(buf.getvalue()).decode()]}]}
        if self._model.startswith("qwen3"):
            body["think"] = False
        for attempt in range(4):  # Ollama answers 500 when a runner restarts; give it a moment
            r = requests.post(f"{self._base}/api/chat", json=body, timeout=600)
            if r.status_code < 500:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        return {"text": r.json()["message"]["content"].strip(), "candidates": None}



ENGINES = {
    "surya": SuryaRec,
    # Default engine of the web app (best measured on the gold set, see eval/reports).
    "qwen3vl-fa": lambda: HFVisionLM("qwen3vl-fa", QWEN_FA),
    # Optional LoRA adapters (experimental; scored lower on the gold set).
    "qwen3vl-lora": lambda: HFVisionLM("qwen3vl-lora", QWEN_FA, adapter=os.environ.get("LORA_ADAPTER", "models/qwen3vl-lora")),
    "qwen25vl-master-lora": lambda: HFVisionLM(
        "qwen25vl-master-lora",
        os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
        adapter=os.environ.get("LORA_ADAPTER", "models/lora_persian_htr_master"),
        prompt="متن دست‌نویس فارسی داخل این تصویر را عیناً رونویسی کن:",
    ),
}


def load_engine(name: str) -> Engine:
    if name.startswith("ollama-vision:"):
        model, _, prep = name[len("ollama-vision:"):].partition("@")
        return OllamaVision(name, model, tuple(prep.split(",")) if prep else ())
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return ENGINES[name]()
