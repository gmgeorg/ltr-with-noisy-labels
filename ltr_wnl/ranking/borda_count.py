"""
Borda Count ranking algorithm.
"""

from collections import defaultdict

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.ranking.ranker_base import Ranker


class BordaCountRanker(Ranker):
    """
    Ranks items by total pairwise wins (Borda Count).

    This is a simple, robust baseline ranking method that counts
    how many pairwise comparisons each item won and ranks them
    by their total win count.

    Complexity: O(C) where C is the number of comparisons

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>>
    >>> data = ComparisonData(n_items=3)
    >>> data.add_comparison(0, 1, outcome=1)   # 0 > 1
    >>> data.add_comparison(0, 2, outcome=1)   # 0 > 2
    >>> data.add_comparison(1, 2, outcome=1)   # 1 > 2
    >>>
    >>> ranker = BordaCountRanker()
    >>> ranking = ranker.fit(data)
    >>> ranking
    [0, 1, 2]
    """

    def fit(self, comparison_data: ComparisonData) -> list[int]:
        """
        Infer ranking using Borda Count (total wins).

        Parameters
        ----------
        comparison_data : ComparisonData
            Pairwise comparison outcomes

        Returns
        -------
        ranking : list[int]
            Estimated ranking (item indices from best to worst)
        """
        n_items = comparison_data.n_items
        win_counts = defaultdict(int)

        # Count total wins for each item
        for (i, j), outcomes in comparison_data.comparisons.items():
            for outcome in outcomes:
                if outcome == 1:
                    win_counts[i] += 1
                else:
                    win_counts[j] += 1

        # Create list of (win_count, item_index) and sort
        scores = [(win_counts[i], i) for i in range(n_items)]
        scores.sort(reverse=True)  # Highest wins first

        # Extract ranking (list of item indices)
        ranking = [item_idx for _, item_idx in scores]
        return ranking

    def __repr__(self) -> str:
        return "BordaCountRanker()"
