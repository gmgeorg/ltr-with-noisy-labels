"""
ltr_wnl: Learning to Rank with Noisy Labels

A Python package for simulating and analyzing learning-to-rank algorithms
under Beta-distributed comparison noise.
"""

__version__ = "0.1.0"

from ltr_wnl import (
    comparisons,
    ground_truth,
    metrics,
    noise,
    ranking,
    sampling,
    simulation,
    utils,
)

__all__ = [
    "comparisons",
    "ground_truth",
    "metrics",
    "noise",
    "ranking",
    "sampling",
    "simulation",
    "utils",
]
