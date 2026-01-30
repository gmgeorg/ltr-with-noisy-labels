"""
Abstract base class for sampling strategies.
"""

from abc import ABC, abstractmethod

from ltr_wnl.comparisons.data import ComparisonData


class SamplerStrategy(ABC):
    """
    Abstract base class for pair selection strategies.

    A sampling strategy determines which pair of items to compare next,
    potentially using information from previous comparisons (active learning).
    """

    @abstractmethod
    def select_next_pair(
        self,
        comparison_data: ComparisonData,
        n_items: int
    ) -> tuple[int, int]:
        """
        Select the next pair of items to compare.

        Parameters
        ----------
        comparison_data : ComparisonData
            History of previous comparisons
        n_items : int
            Total number of items

        Returns
        -------
        i, j : tuple[int, int]
            Indices of the two items to compare (i < j)
        """
        pass
