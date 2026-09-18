"""Compatibility shim for the evolution-log contract.

The implementation now lives in ``experiments.log`` (a package split by
concern: ``schema`` / ``builders`` / ``validate`` / ``io``). This module
re-exports it so the long-standing ``experiments.evolution_log`` import path
keeps working unchanged.

New code may import from either path; ``experiments.log`` is preferred.

See ``experiments/log/__init__.py`` for the full on-disk format contract.
"""
from __future__ import annotations

from experiments.log import *  # noqa: F401,F403
from experiments.log import __all__ as _contract_api

__all__ = list(_contract_api)
