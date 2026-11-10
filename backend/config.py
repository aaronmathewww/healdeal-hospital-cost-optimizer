"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    FLASK_ENV = os.getenv("FLASK_ENV", "development").strip().lower()
    SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
    if FLASK_ENV == "production" and (not SECRET_KEY or SECRET_KEY.lower().startswith("replace-")):
        raise RuntimeError("Set a strong SECRET_KEY in the production environment.")
    if not SECRET_KEY:
        SECRET_KEY = secrets.token_urlsafe(32)
    DEBUG = as_bool(os.getenv("FLASK_DEBUG"), False)
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))

    # CORS: restrict to the local app by default; explicitly configure deployed origins.
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")

    # LLM (Gemini)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

    # LLM fallback (Hugging Face Inference API)
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # Runtime limits
    CHAT_TIMEOUT_SECONDS = int(os.getenv("CHAT_TIMEOUT_SECONDS", "30"))
