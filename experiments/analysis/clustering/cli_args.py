"""Shared CLI arguments for strategy-clustering entry points."""
from __future__ import annotations

from .pipeline import (
    DEFAULT_CODE_EMBEDDING_MODEL,
    DEFAULT_CODE_EMBEDDING_REVISION,
)


def add_clustering_method_args(parser) -> None:
    parser.add_argument(
        "--code-embedding-model",
        default=DEFAULT_CODE_EMBEDDING_MODEL,
        help="local Hugging Face code-embedding model",
    )
    parser.add_argument(
        "--code-embedding-revision",
        default=DEFAULT_CODE_EMBEDDING_REVISION,
        help="pinned Hugging Face model revision used for cache correctness",
    )
    parser.add_argument(
        "--embedding-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="code-embedding device (default: auto; CUDA with CPU fallback)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="maximum code-embedding batch size (default: automatic by token length)",
    )
    parser.add_argument(
        "--embedding-cache",
        action=__import__("argparse").BooleanOptionalAction,
        default=True,
        help="reuse the shared SQLite embedding and DeepSeek-name cache",
    )
    parser.add_argument(
        "--embedding-cache-path",
        default=None,
        help="SQLite cache path (default: results/.analysis_cache/strategy_analysis.sqlite3)",
    )
    parser.add_argument(
        "--refresh-cluster-names",
        action="store_true",
        help="call DeepSeek again even when the same naming request is cached",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="DeepSeek model for semantic cluster names (default: DEEPSEEK_MODEL)",
    )


def clustering_method_kwargs(args) -> dict:
    return {
        "code_embedding_model": args.code_embedding_model,
        "code_embedding_revision": args.code_embedding_revision,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_cache": args.embedding_cache,
        "embedding_cache_path": args.embedding_cache_path,
        "llm_model": args.llm_model,
        "refresh_cluster_names": args.refresh_cluster_names,
        "analysis_source_path": getattr(args, "json", None),
    }
