"""Code-embedding strategy-clustering pipeline.

Complete code is embedded by a local code-specific transformer, then processed
with silhouette-based K selection, K-means, centered PCA, and DeepSeek naming.
``clustering.io.load_generations`` produces the generation records consumed by
``cluster_strategies``.
"""
from __future__ import annotations

import json
import re
import gc
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from experiments.config.load_env import (
    get_base_url,
    get_model,
    require_api_key,
)
from experiments.evolution_log import (
    F_AGENT_ID, F_CODE, F_GENERATION, F_LINEAGE_ID, F_ORIGIN,
    F_PARENT_ID, F_PARENT_LINEAGE_ID, F_POPULATION,
)

from .cache import AnalysisCache, code_hash, stable_hash
from .comments import strip_code_comments

DEFAULT_CODE_EMBEDDING_MODEL = "Salesforce/SFR-Embedding-Code-400M_R"
DEFAULT_CODE_EMBEDDING_REVISION = "cb950dc80d677c6fdc00f56c8ddd20ca2642c59e"
CODE_EMBEDDING_MAX_TOKENS = 8192
EMBEDDING_CACHE_VERSION = 1
CLUSTER_NAMING_PROMPT_VERSION = 1
LEAF_NAMING_PROMPT_VERSION = 1
CODE_CHUNK_OVERLAP = 256
CODE_AGGREGATION_VERSION = "full-code-token-weighted-v1"
CLUSTERING_METHODS = ("kmeans", "hierarchical")


class _HierarchicalClusterModel:
    """KMeans-compatible adapter around :class:`AgglomerativeClustering`.

    Agglomerative clustering exposes ``labels_``/``n_clusters`` but neither
    ``cluster_centers_`` nor ``predict``. This wrapper derives per-cluster
    centroids as the mean of their members and predicts by nearest centroid,
    so downstream pipeline code (representative selection, shared-state
    re-projection) works unchanged.

    The full merge tree is preserved on ``children_`` / ``distances_``
    (same semantics as sklearn's attributes) and can be converted to a
    scipy linkage matrix via :meth:`linkage_matrix` for dendrograms.
    """

    def __init__(self, X, labels, n_clusters, *, children=None, distances=None):
        self.labels_ = np.asarray(labels)
        self.n_clusters = n_clusters
        self._centers = np.vstack(
            [X[self.labels_ == c].mean(axis=0) for c in range(n_clusters)]
        )
        self.children_ = None if children is None else np.asarray(children, dtype=int)
        self.distances_ = None if distances is None else np.asarray(distances, dtype=float)

    @property
    def cluster_centers_(self):
        return self._centers

    def predict(self, X):
        X = np.asarray(X)
        if hasattr(X, "toarray"):
            X = X.toarray()
        distances = np.linalg.norm(
            X[:, None, :] - self._centers[None, :, :], axis=2
        )
        return np.argmin(distances, axis=1)

    def linkage_matrix(self) -> np.ndarray:
        """Return a scipy-compatible linkage matrix from the merge tree.

        sklearn's ``children_`` / ``distances_`` already use scipy's node
        numbering (leaves 0..n-1, internal nodes n..2n-2), so the matrix is
        ``[left, right, distance, member_count]`` per merge.
        """
        if self.children_ is None or self.distances_ is None:
            raise RuntimeError(
                "linkage_matrix() requires a hierarchical fit with the merge "
                "tree retained; re-run fit_clusterer(method='hierarchical')"
            )
        n_samples = self.children_.shape[0] + 1
        return _linkage_from_children(
            self.children_, self.distances_, n_samples
        )


def _linkage_from_children(children, distances, n_samples: int) -> np.ndarray:
    """Build a scipy linkage matrix from sklearn's merge tree arrays."""
    from scipy.cluster.hierarchy import linkage as _validate  # noqa: F401

    children = np.asarray(children, dtype=int)
    distances = np.asarray(distances, dtype=float)
    member_counts = np.ones(n_samples + len(children), dtype=int)
    for i, (left, right) in enumerate(children):
        member_counts[n_samples + i] = member_counts[left] + member_counts[right]
    linkage = np.zeros((len(children), 4), dtype=float)
    linkage[:, 0] = children[:, 0]
    linkage[:, 1] = children[:, 1]
    linkage[:, 2] = distances
    linkage[:, 3] = member_counts[n_samples : n_samples + len(children)]
    return linkage


def _cut_tree(children, distances, n_samples: int, k: int) -> np.ndarray:
    """Cut a full merge tree into exactly ``k`` flat clusters.

    Reuses the scipy linkage matrix + ``fcluster(criterion="maxclust")``,
    which is the canonical way to cut a dendrogram at a fixed cluster count.
    """
    if k <= 1:
        return np.zeros(n_samples, dtype=int)
    if k >= n_samples:
        return np.arange(n_samples, dtype=int)
    from scipy.cluster.hierarchy import fcluster

    linkage = _linkage_from_children(children, distances, n_samples)
    labels = fcluster(linkage, t=k, criterion="maxclust") - 1  # 1-based -> 0-based
    return labels.astype(int)


def fit_clusterer(X, k: int, seed: int, method: str = "kmeans"):
    """Fit a partition into ``k`` clusters with the requested algorithm.

    ``method`` is one of :data:`CLUSTERING_METHODS`. The returned object
    always provides ``labels_``, ``n_clusters``, ``cluster_centers_`` and
    ``predict`` so callers are agnostic to the underlying algorithm.
    """
    if method == "hierarchical":
        # Build the full merge tree once (distance_threshold=0 keeps every
        # merge), then cut it at the requested K. sklearn 1.9 only exposes
        # ``distances_`` in distance_threshold mode, which is exactly the
        # data a dendrogram needs.
        model = AgglomerativeClustering(
            n_clusters=None, linkage="ward", distance_threshold=0
        )
        model.fit(X)
        labels = _cut_tree(
            model.children_, model.distances_, X.shape[0], k
        )
        return _HierarchicalClusterModel(
            X,
            labels,
            k,
            children=model.children_,
            distances=model.distances_,
        )
    if method != "kmeans":
        raise ValueError(
            f"unknown clustering method {method!r}; "
            f"choose from {CLUSTERING_METHODS}"
        )
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)


def _resolve_embedding_device(device: str) -> str:
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("embedding device must be one of: auto, cuda, cpu")
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --embedding-device cpu "
            "or install the project's CUDA PyTorch build"
        )
    return device


@lru_cache(maxsize=4)
def _load_code_embedding_model(model_name: str, revision: str, device: str):
    """Load and cache a local Sentence Transformers code embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Code embedding requires sentence-transformers. Run `uv sync` to "
            "install the project dependencies."
        ) from exc
    model_kwargs = {}
    if device == "cuda":
        import torch

        # RTX 20-series supports fast FP16 but not native BF16.
        model_kwargs["torch_dtype"] = torch.float16
    load_kwargs = {
        "revision": revision or None,
        "trust_remote_code": True,
        "device": device,
        "model_kwargs": model_kwargs,
    }
    try:
        # A pinned revision already present locally needs no Hub round-trip.
        model = SentenceTransformer(model_name, local_files_only=True, **load_kwargs)
    except OSError:
        model = SentenceTransformer(model_name, **load_kwargs)
    # The SFR checkpoint config supports 8192 positions. Its tokenizer advertises
    # a larger generic limit, so set the actual model limit explicitly.
    if model_name == DEFAULT_CODE_EMBEDDING_MODEL:
        model.max_seq_length = CODE_EMBEDDING_MAX_TOKENS
    return model


def _embedding_config(model_name: str, revision: str, device: str) -> dict:
    return {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "model": model_name,
        "revision": revision,
        "backend": "sentence-transformers-pytorch",
        "device": device,
        "compute_dtype": "float16" if device == "cuda" else "float32",
        "max_tokens": CODE_EMBEDDING_MAX_TOKENS,
        "chunk_overlap": CODE_CHUNK_OVERLAP,
        "aggregation": CODE_AGGREGATION_VERSION,
        "normalize_embeddings": True,
        "input": "complete-original-code",
    }


def _batch_for_length(token_count: int, device: str, maximum: int | None) -> int:
    limits = (
        ((512, 32), (1024, 16), (2048, 8), (4096, 2), (8192, 1))
        if device == "cuda"
        else ((512, 8), (1024, 4), (2048, 2), (8192, 1))
    )
    size = next(batch for limit, batch in limits if token_count <= limit)
    return min(size, maximum) if maximum is not None else size


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _encode_prepared(model, texts, lengths, *, device: str, maximum_batch: int | None):
    """Length-bucket prepared texts and retry CUDA OOM with smaller batches."""
    order = sorted(range(len(texts)), key=lengths.__getitem__)
    output = [None] * len(texts)
    start = 0
    while start < len(order):
        batch_size = _batch_for_length(lengths[order[start]], device, maximum_batch)
        ids = order[start : start + batch_size]
        safe_size = _batch_for_length(
            max(lengths[i] for i in ids), device, maximum_batch
        )
        ids = ids[:safe_size]
        while True:
            try:
                values = model.encode(
                    [texts[i] for i in ids],
                    batch_size=len(ids),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                for i, value in zip(ids, np.asarray(values, dtype=np.float32)):
                    output[i] = value
                start += len(ids)
                break
            except RuntimeError as exc:
                if device != "cuda" or not _is_cuda_oom(exc) or len(ids) == 1:
                    raise
                import torch

                torch.cuda.empty_cache()
                ids = ids[: max(1, len(ids) // 2)]
    return np.vstack(output)


def _prepare_complete_code(model, codes: list[str]):
    """Tokenize every byte-complete input, splitting rather than truncating."""
    tokenizer = model.tokenizer
    max_content = CODE_EMBEDDING_MAX_TOKENS - 2
    texts, owners, weights = [], [], []
    token_counts, chunk_counts = [], []
    for owner, code in enumerate(codes):
        token_ids = tokenizer.encode(code, add_special_tokens=False)
        token_counts.append(len(token_ids))
        if len(token_ids) <= max_content:
            texts.append(code)
            owners.append(owner)
            weights.append(max(1, len(token_ids)))
            chunk_counts.append(1)
            continue
        chunks = 0
        position = 0
        while position < len(token_ids):
            chunk = token_ids[position : position + max_content]
            texts.append(
                tokenizer.decode(
                    chunk, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            owners.append(owner)
            weights.append(len(chunk) if chunks == 0 else max(1, len(chunk) - CODE_CHUNK_OVERLAP))
            chunks += 1
            if position + max_content >= len(token_ids):
                break
            position += max_content - CODE_CHUNK_OVERLAP
        chunk_counts.append(chunks)
    lengths = [min(count, max_content) for count in (
        len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
    )]
    return texts, owners, weights, lengths, token_counts, chunk_counts


def _aggregate_chunks(values, owners, weights, code_count: int):
    result = np.zeros((code_count, values.shape[1]), dtype=np.float32)
    totals = np.zeros(code_count, dtype=np.float64)
    for value, owner, weight in zip(values, owners, weights):
        result[owner] += value * weight
        totals[owner] += weight
    result /= totals[:, None]
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1e-12)


def embed_codes(
    codes: list[str],
    *,
    code_embedding_model: str = DEFAULT_CODE_EMBEDDING_MODEL,
    code_embedding_revision: str = DEFAULT_CODE_EMBEDDING_REVISION,
    embedding_device: str = "auto",
    embedding_batch_size: int | None = None,
    embedding_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
):
    """Embed complete code with the local code transformer."""
    if not codes:
        raise ValueError("Cannot embed an empty code collection")
    # Comments and docstrings are documentation, not behavior: strip them so
    # the embedding reflects only executable code. The operation is idempotent
    # and comment-only variants of the same code collapse onto one cached
    # embedding (see ``comments.strip_code_comments``).
    codes = [strip_code_comments(code) for code in codes]
    if embedding_batch_size is not None and embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")
    device = _resolve_embedding_device(embedding_device)
    revision = code_embedding_revision or "main"
    config = _embedding_config(code_embedding_model, revision, device)
    cache = AnalysisCache(embedding_cache_path) if embedding_cache else None
    code_hashes = cache.put_codes(codes) if cache else [
        __import__("hashlib").sha256(code.encode("utf-8")).hexdigest() for code in codes
    ]
    keys = [stable_hash({"embedding": config, "code_hash": h}) for h in code_hashes]
    cached = cache.get_embeddings(keys) if cache else {}
    missing_indices = [i for i, key in enumerate(keys) if key not in cached]

    model = None
    if missing_indices:
        try:
            model = _load_code_embedding_model(code_embedding_model, revision, device)
            missing_codes = [codes[i] for i in missing_indices]
            prepared = _prepare_complete_code(model, missing_codes)
            texts, owners, weights, lengths, token_counts, chunk_counts = prepared
            chunk_vectors = _encode_prepared(
                model, texts, lengths, device=device,
                maximum_batch=embedding_batch_size,
            )
            encoded = _aggregate_chunks(
                chunk_vectors, owners, weights, len(missing_codes)
            )
        except (RuntimeError, OSError) as exc:
            if embedding_device != "auto" or device != "cuda":
                raise
            print(f"  CUDA embedding failed; retrying on CPU: {exc}")
            _load_code_embedding_model.cache_clear()
            model = None
            gc.collect()
            import torch

            torch.cuda.empty_cache()
            return embed_codes(
                codes, code_embedding_model=code_embedding_model,
                code_embedding_revision=code_embedding_revision,
                embedding_device="cpu", embedding_batch_size=embedding_batch_size,
                embedding_cache=embedding_cache,
                embedding_cache_path=embedding_cache_path,
            )
        cache_rows = []
        for local_index, code_index in enumerate(missing_indices):
            key = keys[code_index]
            value = encoded[local_index]
            cached[key] = value
            if cache:
                cache_rows.append({
                    "key": key, "code_hash": code_hashes[code_index],
                    "config": config, "vector": value,
                    "token_count": token_counts[local_index],
                    "chunk_count": chunk_counts[local_index],
                })
        if cache:
            cache.put_embeddings(cache_rows)

    if cache:
        print(
            f"  embedding cache: {len(codes) - len(missing_indices)} hit, "
            f"{len(missing_indices)} miss · device={device}"
        )
    return np.vstack([cached[key] for key in keys]), model


def project_embeddings(X, *, seed: int = 42):
    """Project dense code embeddings to 2-D with centered PCA."""
    reducer = PCA(n_components=2, random_state=seed).fit(X)
    return reducer.transform(X), reducer, "PCA"


def _choose_k(X, n_unique: int, seed: int, method: str = "kmeans") -> int:
    if n_unique < 3:
        print(f"-> too few unique codes ({n_unique}); using K=1")
        return 1

    best_k, best_s = None, -1.0
    # Search K over [2, min(20, n_unique-1)]. The adaptive range is 2-20:
    # experiments can carry 200-390 unique strategies, so a cap of 8
    # systematically under-clustered (silhouette peak often sits at K≈12-20,
    # e.g. v2 seed0: K=15 gives 0.54 vs K=8's 0.45).
    max_k = min(20, n_unique - 1)
    for kk in range(2, max_k + 1):
        km = fit_clusterer(X, kk, seed, method)
        score = silhouette_score(X, km.labels_)
        print(f"  K={kk}: silhouette={score:.4f}")
        if score > best_s:
            best_k, best_s = kk, score
    chosen = best_k if best_k is not None else 1
    print(f"-> chosen K={chosen} (silhouette={best_s:.4f})")
    return chosen


def _representative_codes(X, labels, km, codes: list[str], per_cluster: int = 2):
    """Return the codes nearest each K-means centroid."""
    representatives: dict[int, list[str]] = {}
    for cluster_id in range(km.n_clusters):
        row_ids = np.flatnonzero(np.asarray(labels) == cluster_id)
        center = np.asarray(km.cluster_centers_[cluster_id]).ravel()
        rows = X[row_ids]
        if hasattr(rows, "toarray"):
            rows = rows.toarray()
        distances = np.sum((np.asarray(rows) - center) ** 2, axis=1)
        nearest = row_ids[np.argsort(distances)[:per_cluster]]
        representatives[cluster_id] = [codes[int(i)] for i in nearest]
    return representatives


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek returned invalid JSON for cluster names: {text}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("DeepSeek cluster-name response must be a JSON object")
    return value


def summarize_cluster_names(
    X,
    labels,
    km,
    codes: list[str],
    *,
    llm_model: str | None = None,
    naming_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
    refresh_cluster_names: bool = False,
) -> dict[int, str]:
    """Send centroid-near representative code in one DeepSeek naming request."""
    representatives = _representative_codes(X, labels, km, codes)
    sections = []
    for cluster_id, samples in representatives.items():
        rendered = []
        for sample_id, code in enumerate(samples, 1):
            rendered.append(f"Representative {sample_id}:\n```python\n{code}\n```")
        sections.append(f"CLUSTER {cluster_id}\n" + "\n".join(rendered))

    resolved_llm_model = get_model("deepseek", llm_model)
    naming_request = {
        "prompt_version": CLUSTER_NAMING_PROMPT_VERSION,
        "llm_model": resolved_llm_model,
        "clusters": {
            str(cluster_id): samples
            for cluster_id, samples in representatives.items()
        },
    }
    naming_key = stable_hash(naming_request)
    cache = AnalysisCache(embedding_cache_path) if naming_cache else None
    if cache and not refresh_cluster_names:
        cached_names = cache.get_cluster_names(naming_key)
        if cached_names is not None:
            print("  cluster-name cache: hit")
            return cached_names

    from openai import OpenAI

    client = OpenAI(
        api_key=require_api_key("deepseek"),
        base_url=get_base_url("deepseek"),
    )
    response = client.chat.completions.create(
        model=resolved_llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You name clusters of evolved game-strategy Python code. "
                    "Infer the shared behavioral strategy in each cluster, not "
                    "surface syntax. Return only JSON in this exact shape: "
                    '{"clusters":{"0":"concise name","1":"concise name"}}. '
                    "Use concise English noun phrases of at most six words."
                ),
            },
            {
                "role": "user",
                "content": "Name every cluster below.\n\n" + "\n\n".join(sections),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned an empty cluster-name response")
    payload = _extract_json_object(content)
    raw_names = payload.get("clusters")
    if not isinstance(raw_names, dict):
        raise RuntimeError("DeepSeek response is missing the 'clusters' object")

    expected = set(range(km.n_clusters))
    try:
        names = {int(key): str(value).strip() for key, value in raw_names.items()}
    except (TypeError, ValueError) as exc:
        raise RuntimeError("DeepSeek returned non-integer cluster ids") from exc
    if set(names) != expected or any(not name for name in names.values()):
        raise RuntimeError(
            f"DeepSeek must name exactly clusters {sorted(expected)}; got {sorted(names)}"
        )
    if cache:
        cache.put_cluster_names(
            key=naming_key,
            llm_model=resolved_llm_model,
            request=naming_request,
            names=names,
        )
        print("  cluster-name cache: stored")
    return names


def summarize_leaf_names(
    codes: list[str],
    *,
    llm_model: str | None = None,
    naming_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
    refresh_cluster_names: bool = False,
    batch_size: int = 20,
    max_code_chars: int = 1200,
) -> dict[str, str]:
    """Ask the LLM for a concise behavioral label per unique strategy code.

    Each leaf (unique strategy) is named individually so a dendrogram can show
    *what each strategy is* instead of just a truncated first line. Codes are
    submitted in small batches to stay within the LLM context window; results
    are cached by code hash in the shared SQLite cache.

    Returns ``{code_text: label}`` for every input code.
    """
    unique = sorted(set(codes))
    cache = AnalysisCache(embedding_cache_path) if naming_cache else None
    cache_hashes = cache.put_codes(unique) if cache else [
        code_hash(code) for code in unique
    ]
    cached = cache.get_leaf_names(cache_hashes) if cache else {}

    resolved_llm_model = get_model("deepseek", llm_model)
    names: dict[str, str] = {}
    missing = []
    for digest, code in zip(cache_hashes, unique):
        if digest in cached:
            names[code] = cached[digest]
        else:
            missing.append(code)
    print(f"  leaf naming: {len(names)} cached, "
          f"{len(missing)} to name · model={resolved_llm_model}")

    from openai import OpenAI

    client = OpenAI(
        api_key=require_api_key("deepseek"),
        base_url=get_base_url("deepseek"),
    )
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        section = "\n\n".join(
            f"STRATEGY {i}:\n```python\n{code[:max_code_chars]}\n```"
            for i, code in enumerate(batch)
        )
        response = client.chat.completions.create(
            model=resolved_llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You name individual evolved game-strategy Python "
                        "functions. Infer each strategy's core behavioral "
                        "logic (how it updates reputation and when it "
                        "cooperates), not surface syntax. Return only JSON "
                        "with integer keys matching each STRATEGY index and "
                        "concise English noun phrases of at most six words, "
                        "e.g. "
                        '{"0":"Conditional reputation cooperator",'
                        '"1":"Fixed threshold defector"}.'
                    ),
                },
                {
                    "role": "user",
                    "content": "Name every strategy below.\n\n" + section,
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek returned an empty leaf-name response")
        payload = _extract_json_object(content)
        raw = payload.get("strategies", payload)
        if not isinstance(raw, dict):
            raise RuntimeError("DeepSeek leaf-name response missing 'strategies'")
        try:
            batch_names = {
                int(key): str(value).strip()
                for key, value in raw.items()
                if str(value).strip()
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DeepSeek returned non-integer leaf indices") from exc
        if set(batch_names) != set(range(len(batch))):
            raise RuntimeError(
                f"DeepSeek must name every strategy in the batch; "
                f"got indices {sorted(batch_names)} for {len(batch)} strategies"
            )
        for i, code in enumerate(batch):
            names[code] = batch_names[i]
        if cache:
            cache.put_leaf_names(
                llm_model=resolved_llm_model,
                prompt_version=LEAF_NAMING_PROMPT_VERSION,
                names={code_hash(code): batch_names[i]
                       for i, code in enumerate(batch)},
            )
    return names


def serialize_id_list(ids) -> str:
    """Render agent ids for an annotation: single id as-is, many ids as a list."""
    ids = sorted(ids)
    if len(ids) == 1:
        return str(ids[0])
    return f"[{', '.join(map(str, ids))}]"


def cluster_codes(
    codes: list[str],
    k: int | None = None,
    seed: int = 42,
    *,
    code_embedding_model: str = DEFAULT_CODE_EMBEDDING_MODEL,
    code_embedding_revision: str = DEFAULT_CODE_EMBEDDING_REVISION,
    embedding_device: str = "auto",
    embedding_batch_size: int | None = None,
    embedding_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
    llm_model: str | None = None,
    refresh_cluster_names: bool = False,
    clustering_method: str = "kmeans",
    name_leaves: bool = False,
    analysis_source_path: str | Path | None = None,
):
    """Deduplicate, embed, cluster, and name strategy code.

    Returns ``(X, labels, km, unique_codes, cluster_names)``. If ``k`` is not
    supplied, silhouette score selects K from ``[2, min(20, n_unique - 1)]``.

    With ``name_leaves=True`` each unique strategy code also receives an
    LLM-generated behavioral label (stored on ``km.leaf_names``), which is
    useful for hierarchical dendrogram leaves.
    """
    unique = sorted(set(codes))
    if analysis_source_path:
        AnalysisCache(embedding_cache_path).put_occurrences([
            {"code": code, "source_path": analysis_source_path}
            for code in unique
        ])
    X, _ = embed_codes(
        unique,
        code_embedding_model=code_embedding_model,
        code_embedding_revision=code_embedding_revision,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        embedding_cache=embedding_cache,
        embedding_cache_path=embedding_cache_path,
    )
    if k is None:
        k = _choose_k(X, len(unique), seed, method=clustering_method)
    if not 1 <= k <= len(unique):
        raise ValueError(f"k must be between 1 and {len(unique)}, got {k}")

    km = fit_clusterer(X, k, seed, clustering_method)
    names = summarize_cluster_names(
        X,
        km.labels_,
        km,
        unique,
        llm_model=llm_model,
        naming_cache=embedding_cache,
        embedding_cache_path=embedding_cache_path,
        refresh_cluster_names=refresh_cluster_names,
    )
    km.leaf_names = None
    if name_leaves:
        km.leaf_names = summarize_leaf_names(
            unique,
            llm_model=llm_model,
            naming_cache=embedding_cache,
            embedding_cache_path=embedding_cache_path,
            refresh_cluster_names=refresh_cluster_names,
        )
    run_id = None
    if analysis_source_path:
        run_id = AnalysisCache(embedding_cache_path).put_clustering_run(
            source_path=analysis_source_path,
            embedding_method="code",
            model_name=code_embedding_model,
            cluster_count=k, seed=seed,
            parameters={"scope": "unique_codes", "requested_k": k},
            cluster_names=names,
            assignments=[{"code": code, "cluster_id": int(label)}
                         for code, label in zip(unique, km.labels_)],
        )
    km.analysis_run_id = run_id
    return X, km.labels_, km, unique, names


def cluster_strategies(
    generations,
    k: int | None = None,
    seed: int = 42,
    *,
    code_embedding_model: str = DEFAULT_CODE_EMBEDDING_MODEL,
    code_embedding_revision: str = DEFAULT_CODE_EMBEDDING_REVISION,
    embedding_device: str = "auto",
    embedding_batch_size: int | None = None,
    embedding_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
    llm_model: str | None = None,
    refresh_cluster_names: bool = False,
    clustering_method: str = "kmeans",
    analysis_source_path: str | Path | None = None,
    shared_state: dict | None = None,
) -> dict:
    """Fit one global embedding + K-means + SVD model over all generations.

    ``shared_state``: when provided (a dict returned by a previous call),
    the KMeans model, PCA reducer and cluster names are reused from it
    instead of being re-fitted. This lets several runs (e.g. different
    seeds of the same experiment) share ONE global clustering so labels
    and cluster names are consistent across runs, while each run keeps
    its own generations/rows for plotting.
    """
    all_codes = [
        agent[F_CODE]
        for generation in generations
        for agent in generation[F_POPULATION]
    ]
    if not all_codes:
        raise ValueError("Cannot cluster generations without strategy code")

    if analysis_source_path:
        occurrence_records = []
        for generation in generations:
            generation_id = generation.get(F_GENERATION)
            for agent in generation[F_POPULATION]:
                occurrence_records.append({
                    "code": agent[F_CODE],
                    "source_path": analysis_source_path,
                    "generation": generation_id,
                    "agent_id": agent.get(F_AGENT_ID),
                    "lineage_id": agent.get(F_LINEAGE_ID),
                    "metadata": {
                        "origin": agent.get(F_ORIGIN),
                        "parent_id": agent.get(F_PARENT_ID),
                        "parent_lineage_id": agent.get(F_PARENT_LINEAGE_ID),
                    },
                })
        AnalysisCache(embedding_cache_path).put_occurrences(occurrence_records)

    # Embed each distinct strategy once, then expand to population rows. This is
    # especially important for transformer inference across many generations.
    unique = sorted(set(all_codes))
    Xu, _ = embed_codes(
        unique,
        code_embedding_model=code_embedding_model,
        code_embedding_revision=code_embedding_revision,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        embedding_cache=embedding_cache,
        embedding_cache_path=embedding_cache_path,
    )
    unique_row = {code: i for i, code in enumerate(unique)}
    full_rows = [unique_row[code] for code in all_codes]
    X = Xu[full_rows]

    if shared_state is not None:
        # Reuse a global clustering: map this run's codes into the shared
        # embedding space, then predict with the shared KMeans and project
        # with the shared PCA. Labels and names stay consistent across runs.
        km = shared_state["km"]
        reducer = shared_state["reducer"]
        projection_label = shared_state["projection_label"]
        cluster_names = shared_state["cluster_names"]
        k = km.n_clusters
        labels_u = km.predict(Xu)
        Zu = reducer.transform(Xu)
    else:
        if k is None:
            k = _choose_k(Xu, len(unique), seed, method=clustering_method)
        if not 1 <= k <= len(unique):
            raise ValueError(f"k must be between 1 and {len(unique)}, got {k}")

        # Cluster and project on UNIQUE strategy codes only. Repeated code
        # occurrences are just abundance in the population; fitting on the
        # full rows would let frequent strategies drag the centroids and
        # PCA axes toward themselves, corrupting the semantic structure.
        km = fit_clusterer(Xu, k, seed, clustering_method)
        labels_u = km.labels_
        Zu, reducer, projection_label = project_embeddings(Xu, seed=seed)

        # Name clusters from the unique strategy codes (km was fit on Xu,
        # so labels_u == km.predict(Xu)).
        cluster_names = summarize_cluster_names(
            Xu,
            labels_u,
            km,
            unique,
            llm_model=llm_model,
            naming_cache=embedding_cache,
            embedding_cache_path=embedding_cache_path,
            refresh_cluster_names=refresh_cluster_names,
        )

    # Expand back to population rows so downstream consumers (gen_rows,
    # per-generation plots) keep their existing row alignment.
    labels = labels_u[full_rows]
    Z = Zu[full_rows]

    run_id = None
    if analysis_source_path:
        assignments = []
        offset = 0
        for generation in generations:
            generation_id = generation.get(F_GENERATION)
            for agent in generation[F_POPULATION]:
                assignments.append({
                    "code": agent[F_CODE], "cluster_id": int(labels[offset]),
                    "generation": generation_id, "agent_id": agent.get(F_AGENT_ID),
                    "lineage_id": agent.get(F_LINEAGE_ID),
                    "experiment_id": agent.get("_analysis_experiment_id"),
                    "source_path": agent.get("_analysis_source_path"),
                })
                offset += 1
        run_id = AnalysisCache(embedding_cache_path).put_clustering_run(
            source_path=analysis_source_path,
            embedding_method="code",
            model_name=code_embedding_model,
            cluster_count=k, seed=seed,
            parameters={"scope": "all_generations", "requested_k": k,
                        "projection": projection_label,
                        "clustering_method": clustering_method},
            cluster_names=cluster_names, assignments=assignments,
        )

    gen_rows = []
    offset = 0
    for generation in generations:
        n_rows = len(generation[F_POPULATION])
        gen_rows.append(list(range(offset, offset + n_rows)))
        offset += n_rows

    return {
        "X": X,
        "labels": labels,
        "Z": Z,
        "km": km,
        "reducer": reducer,
        "cluster_names": cluster_names,
        "gen_rows": gen_rows,
        "all_codes": all_codes,
        "embedding_method": "code",
        "embedding_label": "Code embedding",
        "clustering_method": clustering_method,
        "code_embedding_model": code_embedding_model,
        "projection_label": projection_label,
        "projection_explained_variance_ratio": reducer.explained_variance_ratio_,
        "analysis_run_id": run_id,
        "embedding_cache_path": embedding_cache_path,
    }
