"""
Abstract base class for ranking algorithms.
"""

from abc import ABC, abstractmethod

from ltr_wnl.comparisons.data import ComparisonData


class Ranker(ABC):
    """
    Abstract base class for ranking inference algorithms.

    A ranker takes pairwise comparison data and produces an estimated
    ranking of the items.
    """

    @abstractmethod
    def fit(self, comparison_data: ComparisonData) -> list[int]:
        """
        Infer ranking from comparison data.

        Parameters
        ----------
        comparison_data : ComparisonData
            Pairwise comparison outcomes

        Returns
        -------
        ranking : list[int]
            Estimated ranking as a list of item indices,
            ordered from best (rank 1) to worst (rank n)
        """
        pass
