"""Minimal OpenAI-compatible client helpers for the optional remote VLM mode (OCR_ALLOW_REMOTE=1)."""

from __future__ import annotations

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def resolve_openai_base_url(environ: dict[str, str]) -> str:
    return (environ.get("VLM_API_BASE") or DEFAULT_BASE_URL).strip().rstrip("/")


def resolve_openai_api_key(environ: dict[str, str]) -> str:
    return (environ.get("VLM_API_KEY") or environ.get("OPENAI_API_KEY") or "").strip()


def create_openai_client(*, api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)
