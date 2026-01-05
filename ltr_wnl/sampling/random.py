"""
Random (non-adaptive) sampling strategy.
"""

import numpy as np

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.sampling.strategy_base import SamplingStrategy


class RandomSampling(SamplingStrategy):
    """
    Uniform random sampling of pairs (non-adaptive baseline).

    This strategy selects pairs uniformly at random, without considering
    previous comparison outcomes. It serves as a baseline for evaluating
    more sophisticated active learning strategies.

    Expected comparisons to learn ranking: O(N²)

    Parameters
    ----------
    allow_repeats : bool, default=True
        Whether to allow repeated comparisons of the same pair

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>>
    >>> strategy = RandomSampling(allow_repeats=False)
    >>> data = ComparisonData(n_items=5)
    >>> i, j = strategy.select_next_pair(data, n_items=5)
    >>> 0 <= i < j < 5
    True
    """

    def __init__(self, allow_repeats: bool = True):
        self.allow_repeats = allow_repeats

    def select_next_pair(
        self,
        comparison_data: ComparisonData,
        n_items: int
    ) -> tuple[int, int]:
        """
        Randomly select a pair to compare.

        Parameters
        ----------
        comparison_data : ComparisonData
            History of previous comparisons
        n_items : int
            Total number of items

        Returns
        -------
        i, j : tuple[int, int]
            Randomly selected pair (i < j)

        Raises
        ------
        ValueError
            If allow_repeats=False and all pairs have been compared
        """
        if not self.allow_repeats:
            # Select from uncompared pairs only
            compared_pairs = set(comparison_data.get_all_pairs())
            all_pairs = [
                (i, j) for i in range(n_items) for j in range(i + 1, n_items)
            ]
            available_pairs = [p for p in all_pairs if p not in compared_pairs]

            if not available_pairs:
                raise ValueError(
                    "All pairs have been compared. "
                    "Set allow_repeats=True to continue."
                )

            idx = np.random.randint(len(available_pairs))
            return available_pairs[idx]
        else:
            # Random pair with repeats allowed
            i, j = np.random.choice(n_items, size=2, replace=False)
            return tuple(sorted([i, j]))

    def __repr__(self) -> str:
        return f"RandomSampling(allow_repeats={self.allow_repeats})"
