"""Identify final-generation survivors and match them to canonical norms.

Two jobs that the clustering pipeline does not cover:

1. **Survivors.** The shared analysis cache stores *codes*, not provenance:
   ``code_occurrences.generation`` is written as NULL by ``cluster_codes``, and
   the cache has no cooperation data at all. "Which strategies survived the
   final generation, and how cooperative were they?" therefore has to come from
   the ``evolutionary.json`` records, and the cache is used only to locate those
   codes among the global code set.

2. **Norm matching.** The repository defines a *social norm* as an
   ``(observe, decide)`` pair (see ``v2_quantitative.baselines``). This module
   recovers a strategy's truth tables by probing the compiled callables and
   compares them against every canonical baseline, so a survivor can be labelled
   "this is exactly L3" rather than only "this is cooperative".

Probing convention (same one validated in the social-norm analysis):
  * assessment rows, in ``ASSESSMENT_TABLES`` order (GCG, GCB, BCG, BCB, ...):
    ``A_rep = +/-0.5``, ``A_action in {C, D}``, ``B_rep = +/-1.0``,
    ``B_action = 'cooperate'``, ``my_reputation = 0.0``; a row is GOOD when the
    returned reputation moves *up* from ``A_rep``.
  * action rows, in ``ACTION_TABLES`` order (GG, GB, BG, BB):
    ``my_reputation = +/-1.0``, ``opponent_reputation = +/-1.0``.

``A_rep`` is deliberately +/-0.5 so the usual ``+/-1/3`` update never saturates
at the ``[-1, 1]`` clamp, which would make the sign test ambiguous.
"""
from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path

from experiments.evolution_log import (
    F_AGENT_ID,
    F_CODE,
    F_COOPERATION_RATE,
    K_FINAL_POPULATION,
    load_evolution_json,
)
from experiments.v2_quantitative.baselines import (
    ACTION_TABLES,
    ASSESSMENT_TABLES,
    BASELINES,
)
from experiments.v2_quantitative.executor import V2StrategyExecutor

from .comments import strip_code_comments

# Canonical norms whose truth tables are distinct; aliases are folded onto the
# name that keeps the leading-eight reading unambiguous (IS == SC, SS == L3,
# SJ == L6 -- see baselines.py).
NORM_ALIASES = {"SC": "IS", "SS": "L3", "SJ": "L6"}
# ---------------------------------------------------------------------------
# Survivors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Survivor:
    """One agent in a run's ``final_population``."""

    run_label: str
    run_path: str
    agent_id: int
    code: str
    cooperation_rate: float

    @property
    def seed(self) -> str:
        """Best-effort seed tag for legends (``seed3`` when the label has one)."""
        tail = self.run_label.rsplit("_", 1)[-1]
        return tail if tail.startswith("seed") else self.run_label


def load_final_survivors(paths: list[str | Path]) -> list[Survivor]:
    """Read ``final_population`` from each evolution-log record.

    Raises if a record is unreadable, so a silently-empty survivor set cannot
    turn the highlight layer into a no-op.
    """
    survivors: list[Survivor] = []
    for path in sorted(str(p) for p in paths):
        data = load_evolution_json(Path(path))
        final = data.get(K_FINAL_POPULATION) or []
        if not final:
            raise ValueError(f"{path} has an empty final_population")
        label = Path(path).parent.name
        for agent in final:
            survivors.append(
                Survivor(
                    run_label=label,
                    run_path=str(path),
                    agent_id=int(agent[F_AGENT_ID]),
                    code=str(agent[F_CODE]),
                    cooperation_rate=float(agent.get(F_COOPERATION_RATE, float("nan"))),
                )
            )
    if not survivors:
        raise ValueError("no final-generation survivors found in the given records")
    return survivors


def discover_latest_family(
    root: str | Path, pattern: str = "*/evolutionary.json"
) -> list[Path]:
    """Return the seed records of the most recently written experiment family.

    "Family" = every record whose parent directory shares the label of the
    newest record with its trailing ``_seed<N>`` component removed. This keeps
    the default honest (it is the newest finished experiment) while still
    letting the caller override it with an explicit glob. The pattern is one
    level deep on purpose: the results tree holds tens of thousands of files
    and a recursive glob costs seconds for no gain.
    """
    root = Path(root)
    record_paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=False)]
    if not record_paths:
        raise FileNotFoundError(f"no evolutionary.json found under {root}")
    newest = max(record_paths, key=lambda p: p.stat().st_mtime)

    def family_key(label: str) -> str:
        head, _, tail = label.rpartition("_seed")
        return head if head and tail.isdigit() else label

    key = family_key(newest.parent.name)
    family = [p for p in record_paths if family_key(p.parent.name) == key]
    return sorted(family)


# ---------------------------------------------------------------------------
# Norm matching
# ---------------------------------------------------------------------------

_GOOD_SCORES = (0.5, -0.5)
_PARTNER_SCORES = (1.0, -1.0)
_ACTIONS = ("cooperate", "defect")


def assessment_table(executor: V2StrategyExecutor) -> str:
    """Recover the 8-row assessment table (``G``/``B``) by probing ``observe``."""
    rows = []
    for action in _ACTIONS:
        for actor_rep in _GOOD_SCORES:
            for partner_rep in _PARTNER_SCORES:
                new_rep = executor.observe(
                    actor_rep, action, partner_rep, _ACTIONS[0], 0.0
                )
                rows.append("G" if new_rep > actor_rep else "B")
    return "".join(rows)


def action_table(executor: V2StrategyExecutor) -> str:
    """Recover the 4-row action table (``C``/``D``) by probing ``decide``."""
    return "".join(
        "C" if executor.decide(self_rep, opponent_rep) else "D"
        for self_rep in _PARTNER_SCORES
        for opponent_rep in _PARTNER_SCORES
    )


def canonical_norm_tables() -> dict[str, tuple[str, str]]:
    """``{norm name: (assessment, action)}`` for every distinct canonical norm.

    The leading eight declare their tables as constants, so those are taken
    verbatim. The remaining baselines (ALLC, ALLD, IS, SH) have no declared
    table and are recovered by probing. Synonyms collapse onto one entry
    (``SC`` -> ``IS``, ``SS`` -> ``L3``, ``SJ`` -> ``L6``) so a strategy that
    matches, say, IS is not reported twice.
    """
    tables: dict[str, tuple[str, str]] = {
        name: (ASSESSMENT_TABLES[name], ACTION_TABLES[name])
        for name in sorted(ASSESSMENT_TABLES)
    }
    for name in sorted(BASELINES):
        if name in NORM_ALIASES or name in tables:
            continue
        tables[name] = _probe_tables(BASELINES[name])
    return tables


def _probe_tables(code: str) -> tuple[str, str]:
    executor = V2StrategyExecutor(code)
    return assessment_table(executor), action_table(executor)


def match_norm(code: str, tables: dict[str, tuple[str, str]] | None = None):
    """Return the canonical norm name whose tables ``code`` matches exactly.

    Returns ``None`` for a rule that is not a canonical norm (the common case
    for evolved strategies), and ``"?"`` when the code cannot be compiled.
    """
    tables = tables or canonical_norm_tables()
    try:
        probed = _probe_tables(code)
    except Exception:  # noqa: BLE001 - LLM output can be malformed
        return "?"
    for name, declared in tables.items():
        if probed == declared:
            return name
    return None


def verify_canonical_tables() -> list[str]:
    """Self-check: recovered tables must equal the declared leading-eight ones.

    Returns a list of mismatch descriptions (empty when everything agrees), so
    the caller can print a guard result rather than silently trusting the probe.
    """
    problems = []
    for name in sorted(ASSESSMENT_TABLES):
        assessed, acted = _probe_tables(BASELINES[name])
        declared = (ASSESSMENT_TABLES[name], ACTION_TABLES[name])
        if (assessed, acted) != declared:
            problems.append(f"{name}: probed {(assessed, acted)} != declared {declared}")
    return problems


# ---------------------------------------------------------------------------
# Cluster naming from a random sample of member codes
# ---------------------------------------------------------------------------


def sample_cluster_codes(
    labels, codes: list[str], per_cluster: int, seed: int = 42
) -> dict[int, list[str]]:
    """Pick up to ``per_cluster`` member codes per cluster, deterministically."""
    rng = random.Random(seed)
    buckets: dict[int, list[str]] = {}
    for code, label in zip(codes, labels):
        buckets.setdefault(int(label), []).append(code)
    sampled = {}
    for cluster_id, members in sorted(buckets.items()):
        unique = sorted(set(members))
        if len(unique) <= per_cluster:
            sampled[cluster_id] = unique
        else:
            sampled[cluster_id] = sorted(rng.sample(unique, per_cluster))
    return sampled


def _batched_clusters(
    samples: dict[int, list[str]], budget_chars: int, max_code_chars: int
) -> list[dict[int, list[str]]]:
    """Group clusters into requests that stay under a prompt-size budget.

    Sampling 50 codes per cluster is what makes the name robust to outliers, but
    at K=20 that is ~1000 codes -- far past a single request's context window.
    Clusters are therefore packed greedily until the next one would overflow.
    """
    batches: list[dict[int, list[str]]] = []
    current: dict[int, list[str]] = {}
    current_size = 0
    for cluster_id, members in sorted(samples.items()):
        cost = sum(len(code[:max_code_chars]) for code in members) + 200
        if current and current_size + cost > budget_chars:
            batches.append(current)
            current, current_size = {}, 0
        current[cluster_id] = members
        current_size += cost
    if current:
        batches.append(current)
    return batches


def name_clusters_from_samples(
    samples: dict[int, list[str]],
    *,
    llm_model: str | None = None,
    cache=None,
    prompt_version: int = 2,
    max_code_chars: int = 600,
    request_budget_chars: int = 120_000,
    use_llm: bool = True,
) -> dict[int, str]:
    """Ask the LLM to name each cluster from a *random sample* of its members.

    This is deliberately different from ``pipeline.summarize_cluster_names``,
    which feeds the few codes nearest the centroid: a large global cluster can
    have a centroid neighbourhood that is unrepresentative of its bulk, so this
    variant summarizes a uniform random sample instead. Requests are cached in
    the shared SQLite cache under a distinct ``prompt_version`` marker (the
    pipeline's own request dicts do not carry one), so the two naming schemes
    cannot collide on a cache key.

    Clusters are split across several requests to stay within the model's
    context window; each batch is cached independently, so a failure part-way
    through does not throw away the names already obtained.
    """
    if not use_llm:
        return {cid: "unnamed" for cid in samples}

    from experiments.config.load_env import get_base_url, get_model, require_api_key
    from experiments.analysis.clustering.pipeline import _extract_json_object
    from experiments.analysis.clustering.cache import stable_hash

    resolved_model = get_model("deepseek", llm_model)
    batches = _batched_clusters(samples, request_budget_chars, max_code_chars)
    print(f"  cluster naming: {len(samples)} clusters in {len(batches)} request(s)")

    client = None
    names: dict[int, str] = {}
    for batch_index, batch in enumerate(batches, 1):
        request = {
            "sampled_cluster_namer": prompt_version,
            "llm_model": resolved_model,
            "cluster_ids": sorted(batch),
            "samples": {str(k): v for k, v in batch.items()},
        }
        key = stable_hash(request)
        if cache is not None:
            cached = cache.get_cluster_names(key)
            if cached is not None:
                print(f"  batch {batch_index}/{len(batches)}: cache hit")
                names.update(cached)
                continue

        sections = []
        for cluster_id, members in sorted(batch.items()):
            rendered = "\n\n".join(
                f"Sample {i}:\n```python\n{code[:max_code_chars]}\n```"
                for i, code in enumerate(members, 1)
            )
            sections.append(f"CLUSTER {cluster_id}\n{rendered}")

        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=require_api_key("deepseek"), base_url=get_base_url("deepseek")
            )
        response = client.chat.completions.create(
            model=resolved_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You name clusters of evolved game-strategy Python code from "
                        "a uniform random sample of each cluster's members. Infer the "
                        "shared behavioural strategy (how reputation is updated and "
                        "when cooperation happens), not surface syntax. Some samples "
                        "may be outliers or truncated; name the dominant commonality. "
                        'Return only JSON as {"clusters":{"0":"concise name"}}. Use '
                        "concise English noun phrases of at most six words."
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
        raw = payload.get("clusters")
        if not isinstance(raw, dict):
            raise RuntimeError("DeepSeek response is missing the 'clusters' object")
        batch_names = {int(k): str(v).strip() for k, v in raw.items()}
        expected = set(batch)
        if set(batch_names) != expected or any(not v for v in batch_names.values()):
            raise RuntimeError(
                f"DeepSeek must name exactly clusters {sorted(expected)}; "
                f"got {sorted(batch_names)}"
            )
        if cache is not None:
            cache.put_cluster_names(
                key=key, llm_model=resolved_model, request=request, names=batch_names
            )
        names.update(batch_names)
        print(f"  batch {batch_index}/{len(batches)}: {len(batch)} clusters named")
    return names


def summarise_survivors(survivors: list[Survivor], min_cooperation: float) -> dict:
    """Counts for the console summary and the figure caption.

    Only pass survivors that are actually present in the clustered code set:
    a count that includes points the figure cannot draw is a lie about the
    figure.
    """
    high = [s for s in survivors if s.cooperation_rate >= min_cooperation]
    return {
        "runs": len({s.run_label for s in survivors}),
        "individuals": len(survivors),
        "unique_codes": len({s.code for s in survivors}),
        "high_cooperation_individuals": len(high),
        "high_cooperation_unique_codes": len({s.code for s in high}),
        "min_cooperation": min((s.cooperation_rate for s in survivors), default=float("nan")),
        "max_cooperation": max((s.cooperation_rate for s in survivors), default=float("nan")),
    }


def match_survivors_to_cache(
    survivors: list[Survivor], cache_codes: set[str]
) -> tuple[list[Survivor], list[Survivor]]:
    """Split survivors by whether the figure can actually plot them.

    The cache holds comment-stripped code (``embed_codes`` strips before
    ``put_codes``), so survivor code from an evolution record is compared in
    stripped form. A run whose strategies were never fed to the clustering
    pipeline has almost no overlap, and drawing only its handful of incidental
    matches while reporting the full population would misstate the figure.

    Returns ``(drawable, missing)``.
    """
    drawable, missing = [], []
    for survivor in survivors:
        if strip_code_comments(survivor.code) in cache_codes:
            drawable.append(survivor)
        else:
            missing.append(survivor)
    return drawable, missing


def survivor_coverage(survivors: list[Survivor], cache_codes: set[str]) -> float:
    """Fraction of a run's distinct final codes that exist in the cache."""
    unique = {strip_code_comments(s.code) for s in survivors}
    if not unique:
        return 0.0
    return len(unique & cache_codes) / len(unique)


def discover_latest_archived_family(
    root: str | Path,
    cache_codes: set[str],
    *,
    min_coverage: float = 0.5,
    max_records: int = 80,
    pattern: str = "*/evolutionary.json",
) -> tuple[list[Path], float]:
    """Newest experiment family whose final strategies are in the cache.

    Picking purely by mtime is fragile: an observability or baseline run can be
    newer yet were never passed through the clustering pipeline, so its codes
    are absent and the survivor overlay would highlight almost nothing. Families
    are therefore ranked by recency and the newest one clearing
    ``min_coverage`` wins; if none clears it, the newest family is still
    returned (with its coverage) so the caller can report the problem instead of
    failing silently.

    Returns ``(record paths, coverage)``.
    """
    root = Path(root)
    record_paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=False)]
    if not record_paths:
        raise FileNotFoundError(f"no evolutionary.json found under {root}")
    record_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    def family_key(label: str) -> str:
        head, _, tail = label.rpartition("_seed")
        return head if head and tail.isdigit() else label

    families: dict[str, list[Path]] = {}
    for path in record_paths[:max_records]:
        families.setdefault(family_key(path.parent.name), []).append(path)

    # Insertion order is recency order (``record_paths`` is sorted newest-first),
    # so the scan stops at the first qualifying family instead of parsing every
    # record in the results tree.
    newest: tuple[list[Path], float] | None = None
    for key, paths in families.items():
        try:
            survivors = load_final_survivors(paths)
        except Exception as exc:  # noqa: BLE001 - unreadable record: skip, don't abort
            print(f"  family {key}: skipped ({exc})")
            continue
        coverage = survivor_coverage(survivors, cache_codes)
        if newest is None:
            newest = (paths, coverage)
        if coverage >= min_coverage:
            return sorted(paths), coverage
    if newest is None:
        raise ValueError(f"no readable experiment family under {root}")
    print(
        f"  no family reaches {min_coverage:.0%} cache coverage; "
        f"falling back to the newest ({newest[1]:.0%})"
    )
    return sorted(newest[0]), newest[1]


def load_norm_cache(path: str | Path) -> dict:
    """Read a previously written survivor/norm sidecar, if present."""
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
