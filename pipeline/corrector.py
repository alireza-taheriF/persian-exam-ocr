"""
Domain-aware contextual post-correction module with 426K Persian vocabulary.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import requests

from pipeline.normalizer import ZWNJ, normalize_text
from training.eval.metrics import levenshtein

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
_FREQ_LIST = ROOT / "htr_eval" / "data" / "fa_50k.txt"
_DB_WORDS = ROOT / "htr_eval" / "data" / "persian_words_database.txt"
_EXT_TRAIN = ROOT / "dataset" / "external" / "hw_lines" / "train.jsonl"
_TRAIN_JSONL = ROOT / "dataset" / "train.jsonl"

_PUNCT = re.compile(r"[^\w]|_", re.UNICODE)
_DIGITS = re.compile(r"[0-9۰-۹٠-٩]")
_LATIN = re.compile(r"[A-Za-z]")

_LEXICON: set[str] | None = None
MAX_EDIT_RATIO = 0.5

SYSTEM_PROMPT = (
    "You post-correct OCR output of Persian handwriting. You get JSON with OCR lines and may fix single words.\n"
    "Rules:\n"
    "- Fix a word only when the context clearly demands it: OCR letter confusions (ق/ف/غ, ج/چ/ح/خ, ب/پ/ت/ث/ن/ی), "
    "wrong or missing dots, and misspellings of a real Persian word.\n"
    "- Never add, remove, translate or paraphrase content. Keep the number of words the same.\n"
    "- Keep all numbers and symbols exactly as they are. Never write Latin letters or digits that are not in the input line.\n"
    "- If unsure, leave the word unchanged. Unreadable fragments stay as they are.\n"
    'Return only JSON: {"lines":[{"id":<id>,"text":"<corrected line>"}]} with exactly one entry per input line.'
)


def key(word: str) -> str:
    return _PUNCT.sub("", normalize_text(word).replace(ZWNJ, "").replace("ً", ""))


def get_lexicon() -> set[str]:
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON

    lex: set[str] = set()
    if _DB_WORDS.exists():
        for line in _DB_WORDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lex.add(key(line.strip()))

    if _EXT_TRAIN.exists():
        with open(_EXT_TRAIN, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    for w in (json.loads(l).get("text") or "").split():
                        lex.add(key(w))

    if _FREQ_LIST.exists():
        for line in _FREQ_LIST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lex.add(key(line.split()[0]))

    lex.discard("")
    _LEXICON = lex
    logger.info("Loaded grand Persian lexicon: %d words", len(_LEXICON))
    return _LEXICON


def apply_guards(orig: str, new: str, lex: set[str]) -> str:
    o, n = orig.split(), new.split()
    if new.strip() == orig.strip() or abs(len(n) - len(o)) > 1:
        return orig

    if (_DIGITS.search(new) and set(_DIGITS.findall(new)) - set(_DIGITS.findall(orig))) or \
            (_LATIN.search(new) and set(_LATIN.findall(new)) - set(_LATIN.findall(orig))):
        return orig

    out: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if op == "equal":
            out += o[i1:i2]
            continue
        if i1 == i2 or j1 == j2:
            out += o[i1:i2]
        elif levenshtein(ka := " ".join(map(key, o[i1:i2])), kb := " ".join(map(key, n[j1:j2]))) > MAX_EDIT_RATIO * max(len(ka), len(kb)):
            out += o[i1:i2]
        elif any(key(w) not in lex for w in n[j1:j2] if key(w) and not _DIGITS.search(w)):
            out += o[i1:i2]
        else:
            out += n[j1:j2]
    return " ".join(out)


def correct_lines(lines: list[str], model: str | None = None, base_url: str | None = None) -> list[str]:
    """Optional LLM post-correction through a local Ollama server; returns the input unchanged if it is unreachable."""
    model = model or os.environ.get("CORRECTOR_MODEL", "qwen3:8b")
    base_url = (base_url or os.environ.get("CORRECTOR_BASE_URL", "http://localhost:11434")).rstrip("/")
    if not lines or os.environ.get("OCR_POST_CORRECT", "1") == "0":
        return lines

    lex = get_lexicon()
    payload_lines = [{"id": i, "text": t} for i, t in enumerate(lines) if t.strip()]
    if not payload_lines:
        return lines

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"context_before": [], "lines": payload_lines, "context_after": []}, ensure_ascii=False)}
    ]

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0}
    }

    try:
        r = requests.post(f"{base_url}/api/chat", json=body, timeout=(3, 60))
        if r.status_code != 200:
            return lines
        data = json.loads(r.json()["message"]["content"])
        corrected_map = {int(x["id"]): str(x["text"]) for x in data.get("lines", [])}
        
        out = []
        for i, orig in enumerate(lines):
            if i in corrected_map:
                out.append(apply_guards(orig, corrected_map[i], lex))
            else:
                out.append(orig)
        return out
    except Exception as exc:
        logger.warning("Post-correction skipped due to error: %s", exc)
        return lines
