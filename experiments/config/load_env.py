"""Centralized environment / API key loading.

Loads variables from the project-root .env file. All experiments should
import API keys from this module rather than reading os.environ directly,
so the source of truth is one place.

Every value lives in .env: a provider named ``X`` reads ``X_API_KEY``,
``X_API_BASE`` and ``X_MODEL``. Nothing is baked into this module, and there
are no name aliases, so adding a provider requires no code change.

Library usage:
    from experiments.config.load_env import get_api_key
    key = get_api_key("deepseek")

    # --provider on the run_*.py entry points resolves the same three values
    # (get_api_key / get_base_url / get_model) from the provider name.

Command-line usage:
    python -m experiments.config.load_env                  # every provider
    python -m experiments.config.load_env <provider> [...]  # just these

Lists providers discovered in the environment, one block each. A provider is
reported when any of its three variables is set, so a missing key shows up as
``configured=False`` rather than disappearing. Keys are masked.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_ROLES = ("API_KEY", "API_BASE", "MODEL")


def _env(provider: str, role: str) -> str:
    """Read the <PROVIDER>_<role> environment variable, stripped."""
    return os.getenv(f"{provider.upper()}_{role}", "").strip()


def get_api_key(provider: str) -> str:
    """Return the API key for a provider, or empty string if missing."""
    return _env(provider, "API_KEY")


def get_base_url(provider: str) -> str:
    """Return the API base URL for a provider (empty if not configured)."""
    return _env(provider, "API_BASE")


def get_model(provider: str, explicit: str | None = None) -> str:
    """Return the model name to use. Explicit arg wins; then <PROVIDER>_MODEL."""
    return explicit or _env(provider, "MODEL")


def require_api_key(provider: str) -> str:
    """Like get_api_key but raises a clear error if missing."""
    key = get_api_key(provider)
    if not key:
        raise RuntimeError(
            f"Missing API key for '{provider}'. "
            f"Set {provider.upper()}_API_KEY in .env or your shell."
        )
    return key


def is_configured(provider: str) -> bool:
    """Whether an API key is available for the given provider."""
    return bool(get_api_key(provider))


def available_providers() -> list[str]:
    """Every provider named by the environment.

    Discovers any ``<PROVIDER>_{API_KEY,API_BASE,MODEL}`` variable, so it also
    lists providers that have a base URL or model but no key yet.
    """
    return sorted({
        name[: -len(role) - 1].lower()
        for name in os.environ
        for role in _ROLES
        if name.upper().endswith(f"_{role}") and name.upper() != f"_{role}"
    })


if __name__ == "__main__":
    print(__doc__)
    for provider in sys.argv[1:] or available_providers():
        configured = is_configured(provider)
        masked = (get_api_key(provider)[:6] + "***") if configured else "(not set)"
        print(f"{provider:12s}  configured={configured}  key={masked}")
        print(f"            base_url={get_base_url(provider) or '(not set)'}")
        print(f"            model={get_model(provider) or '(not set)'}")
