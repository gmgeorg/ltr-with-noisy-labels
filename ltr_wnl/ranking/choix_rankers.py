"""
Ranking algorithms using the choix library.

The choix library provides efficient implementations of:
- Plackett-Luce MLE
- Rank Centrality
- Luce Spectral Ranking
- And more...

See: https://github.com/lucasmaystre/choix
"""

import numpy as np
import choix

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.ranking.ranker_base import Ranker


class ChoixRanker(Ranker):
    """
    Base class for rankers using the choix library.

    Parameters
    ----------
    method : str
        Choix method to use. Options include:
        - 'ilsr_pairwise': Iterative Luce Spectral Ranking (default)
        - 'mm_pairwise': Minorization-Maximization for Plackett-Luce
        - 'rank_centrality': Rank Centrality
    alpha : float
        Regularization parameter for stability (default: 1e-6)
    """

    def __init__(self, method: str = "ilsr_pairwise", alpha: float = 1e-6):
        valid_methods = ["ilsr_pairwise", "mm_pairwise", "rank_centrality"]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {valid_methods}"
            )

        self.method = method
        self.alpha = alpha
        self._strengths: np.ndarray | None = None

    def fit(self, comparison_data: ComparisonData) -> list[int]:
        """
        Infer ranking using choix methods.

        Parameters
        ----------
        comparison_data : ComparisonData
            Pairwise comparison outcomes

        Returns
        -------
        ranking : list[int]
            Estimated ranking (item indices from best to worst)
        """
        # Convert ComparisonData to choix format
        # choix expects a list of (winner, loser) tuples
        pairwise_data = []

        for (i, j), outcomes in comparison_data.comparisons.items():
            for outcome in outcomes:
                if outcome == 1:
                    pairwise_data.append((i, j))  # i beat j
                else:
                    pairwise_data.append((j, i))  # j beat i

        # Estimate parameters using choix
        n_items = comparison_data.n_items

        if self.method == "ilsr_pairwise":
            params = choix.ilsr_pairwise(
                n_items=n_items,
                data=pairwise_data,
                alpha=self.alpha
            )
        elif self.method == "mm_pairwise":
            params = choix.opt_pairwise(
                n_items=n_items,
                data=pairwise_data,
                method="BFGS",  # BFGS optimization method
                alpha=self.alpha
            )
        elif self.method == "rank_centrality":
            params = choix.rank_centrality(
                n_items=n_items,
                data=pairwise_data
            )

        # Store strengths for active learning
        self._strengths = params

        # Convert parameters to ranking (higher param = better rank)
        ranking = np.argsort(params)[::-1].tolist()
        return ranking

    def get_strengths(self) -> np.ndarray:
        """
        Get learned strength parameters.

        Returns
        -------
        strengths : np.ndarray
            Strength parameters s_i for each item

        Raises
        ------
        ValueError
            If fit() has not been called yet
        """
        if self._strengths is None:
            raise ValueError(
                "No strengths available. Call fit() first."
            )
        return self._strengths

    def __repr__(self) -> str:
        return f"ChoixRanker(method='{self.method}')"


class PlackettLuceRanker(ChoixRanker):
    """
    Plackett-Luce Maximum Likelihood Estimation ranker.

    Uses the choix library's implementation of the MM algorithm
    for computing MLE of Plackett-Luce model parameters.

    Parameters
    ----------
    alpha : float, default=1e-6
        L2 regularization strength for stability

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>>
    >>> data = ComparisonData(n_items=3)
    >>> data.add_comparison(0, 1, outcome=1)
    >>> data.add_comparison(0, 2, outcome=1)
    >>> data.add_comparison(1, 2, outcome=1)
    >>>
    >>> ranker = PlackettLuceRanker()
    >>> ranking = ranker.fit(data)
    >>> strengths = ranker.get_strengths()
    """

    def __init__(self, alpha: float = 1e-6):
        super().__init__(method="mm_pairwise", alpha=alpha)


class RankCentralityRanker(ChoixRanker):
    """
    Rank Centrality ranker.

    A spectral method that is robust to noise and efficient
    for large-scale ranking problems.

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>>
    >>> data = ComparisonData(n_items=3)
    >>> data.add_comparison(0, 1, outcome=1)
    >>> data.add_comparison(0, 2, outcome=1)
    >>> data.add_comparison(1, 2, outcome=1)
    >>>
    >>> ranker = RankCentralityRanker()
    >>> ranking = ranker.fit(data)
    """

    def __init__(self):
        super().__init__(method="rank_centrality")
