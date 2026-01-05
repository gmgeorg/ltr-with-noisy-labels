"""
Metrics for evaluating ranking quality.
"""

import numpy as np
from scipy.stats import kendalltau


def kendall_tau_distance(
    ranking_pred: list[int],
    ranking_true: list[int]
) -> int:
    """
    Compute Kendall's tau distance (number of pairwise inversions).

    The Kendall tau distance counts the number of pairs that are
    ordered differently in the two rankings.

    Parameters
    ----------
    ranking_pred : list[int]
        Predicted ranking (item indices in order)
    ranking_true : list[int]
        True ranking (item indices in order)

    Returns
    -------
    inversions : int
        Number of discordant pairs (pairwise inversions)

    Examples
    --------
    >>> # Perfect agreement
    >>> kendall_tau_distance([0, 1, 2], [0, 1, 2])
    0
    >>> # Complete disagreement
    >>> kendall_tau_distance([2, 1, 0], [0, 1, 2])
    3
    >>> # One inversion
    >>> kendall_tau_distance([0, 2, 1], [0, 1, 2])
    1
    """
    if len(ranking_pred) != len(ranking_true):
        raise ValueError("Rankings must have the same length")

    n = len(ranking_true)
    max_inversions = n * (n - 1) / 2

    # Use scipy's kendalltau to compute correlation
    tau, _ = kendalltau(ranking_pred, ranking_true)

    # Convert correlation to distance
    # tau ranges from -1 (complete disagreement) to 1 (perfect agreement)
    # inversions = (1 - tau) * max_inversions / 2
    inversions = (1 - tau) * max_inversions / 2

    return int(np.round(inversions))


def normalized_kendall_tau(
    ranking_pred: list[int],
    ranking_true: list[int]
) -> float:
    """
    Normalized Kendall's tau distance in range [0, 1].

    This is the fraction of pairs that are incorrectly ordered.

    Parameters
    ----------
    ranking_pred : list[int]
        Predicted ranking (item indices in order)
    ranking_true : list[int]
        True ranking (item indices in order)

    Returns
    -------
    normalized_distance : float
        Inversions divided by maximum possible inversions,
        in range [0, 1] where 0 is perfect and 1 is worst

    Examples
    --------
    >>> # Perfect agreement
    >>> normalized_kendall_tau([0, 1, 2], [0, 1, 2])
    0.0
    >>> # Complete disagreement
    >>> normalized_kendall_tau([2, 1, 0], [0, 1, 2])
    1.0
    """
    n = len(ranking_true)
    max_inversions = n * (n - 1) / 2

    inversions = kendall_tau_distance(ranking_pred, ranking_true)

    return inversions / max_inversions


def top_k_accuracy(
    ranking_pred: list[int],
    ranking_true: list[int],
    k: int
) -> float:
    """
    Fraction of top-k items correctly identified.

    Computes the proportion of the true top-k items that appear
    in the predicted top-k.

    Parameters
    ----------
    ranking_pred : list[int]
        Predicted ranking (item indices in order)
    ranking_true : list[int]
        True ranking (item indices in order)
    k : int
        Number of top items to consider

    Returns
    -------
    accuracy : float
        Fraction in range [0, 1]

    Examples
    --------
    >>> # Top-2 accuracy
    >>> top_k_accuracy([0, 1, 2, 3], [0, 1, 2, 3], k=2)
    1.0
    >>> top_k_accuracy([1, 0, 2, 3], [0, 1, 2, 3], k=2)
    1.0
    >>> top_k_accuracy([2, 3, 0, 1], [0, 1, 2, 3], k=2)
    0.0
    """
    if k > len(ranking_true):
        raise ValueError(f"k={k} exceeds number of items={len(ranking_true)}")

    true_top_k = set(ranking_true[:k])
    pred_top_k = set(ranking_pred[:k])

    overlap = len(true_top_k & pred_top_k)

    return overlap / k
