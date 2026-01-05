"""
Functions for generating ground truth item strengths and rankings.
"""

import numpy as np


def generate_btl_strengths(
    n_items: int,
    distribution: str = "normal",
    random_state: int | None = None,
    **dist_kwargs,
) -> np.ndarray:
    """
    Generate latent Bradley-Terry-Luce (BTL) item strengths.

    The BTL model assumes each item has a latent strength score,
    and P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j)) where s_i is
    the strength of item i.

    Parameters
    ----------
    n_items : int
        Number of items
    distribution : str, default='normal'
        Distribution for sampling strengths:
        - 'normal': Normal(loc, scale)
        - 'uniform': Uniform(low, high)
        - 'exponential': Exponential(scale)
    random_state : int, optional
        Random seed for reproducibility
    **dist_kwargs : dict
        Additional arguments for the distribution:
        - normal: loc (default=0), scale (default=1)
        - uniform: low (default=0), high (default=1)
        - exponential: scale (default=1)

    Returns
    -------
    strengths : np.ndarray, shape (n_items,)
        Latent strength scores, sorted in descending order
        (index 0 has highest strength)

    Examples
    --------
    >>> strengths = generate_btl_strengths(10, distribution='normal',
    ...                                     random_state=42)
    >>> strengths.shape
    (10,)
    >>> # Strengths are sorted descending
    >>> np.all(strengths[:-1] >= strengths[1:])
    True
    """
    rng = np.random.RandomState(random_state)

    if distribution == "normal":
        loc = dist_kwargs.get("loc", 0.0)
        scale = dist_kwargs.get("scale", 1.0)
        strengths = rng.normal(loc=loc, scale=scale, size=n_items)

    elif distribution == "uniform":
        low = dist_kwargs.get("low", 0.0)
        high = dist_kwargs.get("high", 1.0)
        strengths = rng.uniform(low=low, high=high, size=n_items)

    elif distribution == "exponential":
        scale = dist_kwargs.get("scale", 1.0)
        strengths = rng.exponential(scale=scale, size=n_items)

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: 'normal', 'uniform', 'exponential'"
        )

    # Sort in descending order (best item first)
    return np.sort(strengths)[::-1]


def strengths_to_ranking(strengths: np.ndarray) -> list[int]:
    """
    Convert strength scores to a ranking.

    Parameters
    ----------
    strengths : np.ndarray, shape (n_items,)
        Item strength scores

    Returns
    -------
    ranking : list[int]
        Item indices sorted by strength (descending)

    Examples
    --------
    >>> strengths = np.array([0.5, 2.1, 1.3, 0.8])
    >>> strengths_to_ranking(strengths)
    [1, 2, 3, 0]
    """
    return np.argsort(strengths)[::-1].tolist()


def generate_fixed_ranking(n_items: int) -> tuple[np.ndarray, list[int]]:
    """
    Generate simple monotonic strengths with known ranking [0, 1, ..., n-1].

    This is useful for testing and when you want a deterministic ground truth.

    Parameters
    ----------
    n_items : int
        Number of items

    Returns
    -------
    strengths : np.ndarray, shape (n_items,)
        Monotonically decreasing strengths
    ranking : list[int]
        The true ranking [0, 1, 2, ..., n_items-1]

    Examples
    --------
    >>> strengths, ranking = generate_fixed_ranking(5)
    >>> ranking
    [0, 1, 2, 3, 4]
    >>> strengths
    array([4., 3., 2., 1., 0.])
    """
    strengths = np.arange(n_items, 0, -1, dtype=float)
    ranking = list(range(n_items))
    return strengths, ranking
