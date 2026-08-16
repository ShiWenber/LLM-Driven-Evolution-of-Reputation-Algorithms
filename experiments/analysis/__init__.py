"""Analysis modules for indirect reciprocity simulation.

This package is a lightweight facade: no submodules are imported here, so
``import experiments.analysis`` has no side effects (it does not trigger the
matplotlib / sklearn imports inside the plotting and clustering modules).
Import the specific module or function you need, e.g.:

    from experiments.analysis.lineage import build_lineage_tree
    from experiments.analysis.clustering.io import load_generations
    from experiments.analysis.clustering.pipeline import cluster_strategies
    from experiments.analysis.plot_lineage import lineage_survival_plot
"""
