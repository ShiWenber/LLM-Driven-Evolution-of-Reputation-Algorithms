"""Lineage (evolutionary-tree) analysis.

Submodules:
  - ``build`` : derive tree data from schema-v4 lineage events (pure
                logic, no plotting / heavy deps).

The strategy clustering this sub-package used to host now lives in
``clustering.pipeline`` (``cluster_codes`` / ``summarize_cluster_names``).

Only ``build_lineage_tree`` is exposed here (lazily, PEP 562 ``__getattr__``)
so that ``import experiments.analysis.lineage`` pulls no heavy dependencies.
Rendering lives in ``experiments.analysis.plot_lineage``.

    from experiments.analysis.lineage import build_lineage_tree
    from experiments.analysis.clustering.pipeline import cluster_codes
"""

__all__ = [
    "build_lineage_tree",
]


def __getattr__(name: str):
    if name == "build_lineage_tree":
        from .build import build_lineage_tree
        return build_lineage_tree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
