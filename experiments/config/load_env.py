"""Centralized environment / API key loading.

Loads variables from the project-root .env file. All experiments should
import API keys from this module rather than reading os.environ directly,
so the source of truth is one place.

Usage:
    from experiments.config.load_env import get_api_key
    key = get_api_key("deepseek")
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _ENV_PATH = _PROJECT_ROOT / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        # .env missing is non-fatal; user may use shell env vars instead
        pass
except ImportError:
    # python-dotenv not installed; fall back to os.environ only
    pass


# Provider -> list of env var names to try, in priority order
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "deepseek":   ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"),
    "openai":     ("OPENAI_API_KEY",),
    "anthropic":  ("ANTHROPIC_API_KEY",),
    "google":     ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}

# Provider -> default base URL
_DEFAULT_BASE_URLS: dict[str, str] = {
    "deepseek":  "https://api.deepseek.com",
    "openai":    "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

# Provider -> default model
_DEFAULT_MODELS: dict[str, str] = {
    "deepseek":  "deepseek-v4-flash",
    "openai":    "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google":    "gemini-1.5-pro",
}


def get_api_key(provider: str) -> str:
    """Return the API key for a provider, or empty string if missing.

    Tries provider-specific env var first, then generic <PROVIDER>_API_KEY.
    """
    provider = provider.lower()
    candidates = _PROVIDER_ENV_VARS.get(provider, (f"{provider.upper()}_API_KEY",))
    for name in candidates:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def get_base_url(provider: str) -> str:
    """Return the API base URL for a provider."""
    provider = provider.lower()
    explicit = os.getenv(f"{provider.upper()}_API_BASE", "").strip()
    if explicit:
        return explicit
    return _DEFAULT_BASE_URLS.get(provider, "")


def get_model(provider: str, explicit: str | None = None) -> str:
    """Return the model name to use. Explicit arg wins; then env var; then default."""
    if explicit:
        return explicit
    provider = provider.lower()
    env_model = os.getenv(f"{provider.upper()}_MODEL", "").strip()
    if env_model:
        return env_model
    return _DEFAULT_MODELS.get(provider, "")


def require_api_key(provider: str) -> str:
    """Like get_api_key but raises a clear error if missing."""
    key = get_api_key(provider)
    if not key:
        raise RuntimeError(
            f"Missing API key for '{provider}'. "
            f"Set one of {list(_PROVIDER_ENV_VARS.get(provider, ()))} in .env or your shell."
        )
    return key


def is_configured(provider: str) -> bool:
    """Whether an API key is available for the given provider."""
    return bool(get_api_key(provider))


if __name__ == "__main__":
    # Quick self-test: `python -m experiments.config.load_env`
    for p in ("deepseek", "openai", "anthropic", "google"):
        configured = is_configured(p)
        masked = (get_api_key(p)[:6] + "***") if configured else "(not set)"
        print(f"{p:10s}  configured={configured}  key={masked}")
        print(f"            base_url={get_base_url(p)}")
        print(f"            model={get_model(p)}")
