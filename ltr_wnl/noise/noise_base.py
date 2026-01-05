"""
Abstract base class for noise models.
"""

from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """
    Abstract base class for comparison noise models.

    A noise model determines the probability of comparison errors
    between pairs of items.
    """

    @abstractmethod
    def sample_error_rate(
        self,
        i: int | None = None,
        j: int | None = None
    ) -> float:
        """
        Sample an error probability for a comparison.

        Parameters
        ----------
        i : int, optional
            Index of first item
        j : int, optional
            Index of second item

        Returns
        -------
        error_rate : float
            Probability of comparison error, in range [0, 1]
        """
        pass
