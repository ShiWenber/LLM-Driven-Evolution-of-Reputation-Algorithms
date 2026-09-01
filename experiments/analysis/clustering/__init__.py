"""Reusable strategy-clustering utilities for analysis plotting scripts.

Sub-package layout:
  - ``io``        : parse generation/population data out of an evolutionary.json
  - ``pipeline``  : code embedding + K-means + centered PCA pipeline

Plotting scripts live one level up, in ``experiments/analysis``, and import
from here instead of re-implementing the pipeline.
"""
