"""
Data structures for storing pairwise comparison results.
"""

from collections import defaultdict

import numpy as np


class ComparisonData:
    """
    Storage and management of pairwise comparison data.

    This class stores comparison outcomes and provides utilities
    for querying and analyzing the comparison history.

    Parameters
    ----------
    n_items : int
        Number of items being compared

    Attributes
    ----------
    n_items : int
        Number of items
    comparisons : dict[tuple[int, int], list[int]]
        Dictionary mapping (i, j) pairs (where i < j) to a list of outcomes.
        Outcome is 1 if i won, -1 if j won.

    Examples
    --------
    >>> data = ComparisonData(n_items=5)
    >>> data.add_comparison(0, 1, outcome=1)  # item 0 beat item 1
    >>> data.add_comparison(1, 0, outcome=-1) # item 1 beat item 0 (stored as 0,1)
    >>> data.get_win_count(0, 1)
    1
    >>> data.get_win_count(1, 0)
    1
    """

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.comparisons: dict[tuple[int, int], list[int]] = defaultdict(list)

    def add_comparison(self, i: int, j: int, outcome: int) -> None:
        """
        Add a single comparison outcome.

        Parameters
        ----------
        i : int
            Index of first item
        j : int
            Index of second item
        outcome : int
            1 if i won, -1 if j won
        """
        if outcome not in {1, -1}:
            raise ValueError(f"outcome must be 1 or -1, got {outcome}")

        # Standardize: always store (i, j) where i < j
        if i < j:
            self.comparisons[(i, j)].append(outcome)
        else:
            # Flip outcome since we're reversing the pair
            self.comparisons[(j, i)].append(-outcome)

    def get_all_pairs(self) -> list[tuple[int, int]]:
        """
        Get all pairs that have been compared.

        Returns
        -------
        pairs : list[tuple[int, int]]
            List of (i, j) pairs where i < j
        """
        return list(self.comparisons.keys())

    def get_outcomes(self, i: int, j: int) -> list[int]:
        """
        Get all comparison outcomes for a specific pair.

        Parameters
        ----------
        i : int
            Index of first item
        j : int
            Index of second item

        Returns
        -------
        outcomes : list[int]
            List of outcomes (1 if i won, -1 if j won)
        """
        if i < j:
            return self.comparisons.get((i, j), [])
        else:
            # Flip outcomes since we're reversing the pair
            return [-o for o in self.comparisons.get((j, i), [])]

    def get_win_count(self, i: int, j: int) -> int:
        """
        Get number of times item i beat item j.

        Parameters
        ----------
        i : int
            Index of first item
        j : int
            Index of second item

        Returns
        -------
        wins : int
            Number of times i beat j
        """
        outcomes = self.get_outcomes(i, j)
        return sum(1 for o in outcomes if o == 1)

    def get_win_matrix(self) -> np.ndarray:
        """
        Get win count matrix W where W[i,j] = number of times i beat j.

        Returns
        -------
        win_matrix : np.ndarray, shape (n_items, n_items)
            Win count matrix
        """
        W = np.zeros((self.n_items, self.n_items), dtype=int)

        for (i, j), outcomes in self.comparisons.items():
            for outcome in outcomes:
                if outcome == 1:
                    W[i, j] += 1
                else:
                    W[j, i] += 1

        return W

    def total_comparisons(self) -> int:
        """
        Get total number of comparisons performed.

        Returns
        -------
        total : int
            Total number of comparisons
        """
        return sum(len(outcomes) for outcomes in self.comparisons.values())

    def __repr__(self) -> str:
        n_pairs = len(self.comparisons)
        n_total = self.total_comparisons()
        return (
            f"ComparisonData(n_items={self.n_items}, "
            f"n_pairs={n_pairs}, total_comparisons={n_total})"
        )
