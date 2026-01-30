"""Gain based sampling techniques.

Picks pairs with highest and lowest gain as a strategy.
"""

import numpy as np
from ltr_wnl.sampling.strategy_base import SamplerStrategy, ComparisonData
from ltr_wnl.ranking.choix_rankers import ChoixRanker, PlackettLuceRanker

class MaxGainSampler(SamplerStrategy):
    """
    Active learning strategy using Fisher Information (D-Optimality).
    
    It selects pairs that maximize the expected information gain for the 
    Bradley-Terry model parameters.

    Uses Plackett-Luce to estimate item strengths, then selects pairs
    with smallest score differences (highest entropy / most uncertain).

    For BTL model P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j)),
    maximum entropy occurs when s_i ≈ s_j (P ≈ 0.5).
    """
    def __init__(
        self, 
        ranker: ChoixRanker, 
        batch_size: int, 
        burn_in_count: int
    ):
        self.ranker = ranker
        self.batch_size = batch_size
        self.burn_in_count = burn_in_count
        
        self._current_batch = []
        self._batch_idx = 0

    def select_next_pair(self, comparison_data, n_items: int) -> tuple[int, int]:
        total_collected = comparison_data.total_comparisons()
        
        # 1. Burn-in Phase: Random sampling to get a baseline model
        if total_collected < self.burn_in_count:
            return self._random_pair(n_items)

        # 2. Refit and Batch Selection
        if self._batch_idx >= len(self._current_batch):
            self._compute_next_batch(comparison_data, n_items)
            self._batch_idx = 0

        # 3. Return from active batch
        if not self._current_batch: # Fallback if batching failed
            return self._random_pair(n_items)
            
        pair = self._current_batch[self._batch_idx]
        self._batch_idx += 1
        return pair

    def _compute_next_batch(self, data, n_items):
        try:
            self.ranker.fit(data)
            strengths = self.ranker.get_strengths()

            # Get already-compared pairs to avoid reusing them
            already_compared = set()
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    if data.get_outcomes(i, j):
                        already_compared.add((i, j))

            # Scores for all UNCOMPARED pairs
            candidate_pairs = []
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    if (i, j) not in already_compared:
                        # Fisher Information weight: p * (1-p)
                        # This is maximized when s_i == s_j
                        prob_i_wins = np.exp(strengths[i]) / (np.exp(strengths[i]) + np.exp(strengths[j]))
                        info_gain = prob_i_wins * (1.0 - prob_i_wins)

                        candidate_pairs.append(((i, j), info_gain))

            # If we've compared everything, allow reuse
            if not candidate_pairs:
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        prob_i_wins = np.exp(strengths[i]) / (np.exp(strengths[i]) + np.exp(strengths[j]))
                        info_gain = prob_i_wins * (1.0 - prob_i_wins)
                        candidate_pairs.append(((i, j), info_gain))

            # Sort by highest gain
            candidate_pairs.sort(key=lambda x: x[1], reverse=True)
            self._current_batch = [p for p, g in candidate_pairs[:self.batch_size]]

        except Exception:
            self._current_batch = [self._random_pair(n_items) for _ in range(self.batch_size)]

    def _random_pair(self, n):
        i, j = np.random.choice(n, size=2, replace=False)
        return tuple(sorted((i, j)))
    

class MinGainSampler(SamplerStrategy):
    """
    Control strategy that selects pairs with lowest uncertainty.

    Opposite of active learning - selects pairs with largest score
    differences (lowest entropy / most certain) to test whether
    model-guided selection matters at all.

    For BTL model P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j)),
    minimum entropy occurs when |s_i - s_j| is large (P → 0 or 1).

    Parameters
    ----------
    ranker : PlackettLuceRanker
        Ranker to use for estimating item strengths
    batch_size : int, default=100
        Number of pairs to collect before refitting model
    burn_in_fraction : float, default=0.2
        Fraction of total comparisons to do randomly before anti-active learning.
        Ignored if burn_in_count is specified.
    burn_in_count : int | None, default=None
        Explicit number of random comparisons before anti-active learning.
        If None, uses batch_size as a proxy for burn-in.
    random_state : int | None, default=None
        Random seed for reproducibility

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>> from ltr_wnl.ranking import PlackettLuceRanker
    >>>
    >>> strategy = AntiActiveSampler(
    ...     ranker=PlackettLuceRanker(),
    ...     batch_size=10,
    ...     burn_in_fraction=0.2
    ... )
    >>> data = ComparisonData(n_items=5)
    >>> # During burn-in: random sampling
    >>> # After burn-in: select pairs with largest |s_i - s_j|
    """

    def __init__(
        self,
        ranker: PlackettLuceRanker,
        batch_size: int = 100,
        burn_in_fraction: float = 0.2,
        burn_in_count: int | None = None,
        random_state: int | None = None
    ):
        self.ranker = ranker
        self.batch_size = batch_size
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_count = burn_in_count
        self.random_state = random_state

        # Internal state
        self._burn_in_complete = False
        self._current_batch_pairs: list[tuple[int, int]] = []
        self._batch_index = 0

        if random_state is not None:
            np.random.seed(random_state)

    def select_next_pair(
        self,
        comparison_data: ComparisonData,
        n_items: int
    ) -> tuple[int, int]:
        """
        Select next pair using anti-active sampling.

        During burn-in: random sampling
        After burn-in: select pairs with largest |s_i - s_j|

        Parameters
        ----------
        comparison_data : ComparisonData
            History of previous comparisons
        n_items : int
            Total number of items

        Returns
        -------
        i, j : tuple[int, int]
            Selected pair (i < j)
        """
        # Check if we need to refit and select new batch
        if self._should_refit(comparison_data):
            self._select_new_batch(comparison_data, n_items)

        # During burn-in or if batch is empty: random sampling
        if not self._burn_in_complete or not self._current_batch_pairs:
            return self._random_pair(n_items)

        # Select next pair from current batch
        pair = self._current_batch_pairs[self._batch_index]
        self._batch_index += 1

        return pair

    def _should_refit(self, comparison_data: ComparisonData) -> bool:
        """Check if we should refit the model and select a new batch."""
        total_comparisons = comparison_data.total_comparisons()

        # First check: just completed burn-in
        if not self._burn_in_complete and total_comparisons > 0:
            # Use explicit burn_in_count if provided, otherwise use batch_size
            burn_in_threshold = self.burn_in_count if self.burn_in_count is not None else self.batch_size

            if total_comparisons >= burn_in_threshold:
                self._burn_in_complete = True
                return True

        # Subsequent checks: exhausted current batch
        if self._burn_in_complete and self._batch_index >= len(self._current_batch_pairs):
            return True

        return False

    def _select_new_batch(self, comparison_data: ComparisonData, n_items: int) -> None:
        """Fit model and select next batch of low-uncertainty pairs."""
        try:
            # Fit ranker to get current strength estimates
            self.ranker.fit(comparison_data)
            strengths = self.ranker.get_strengths()

            # Get already-compared pairs to avoid reusing them
            already_compared = set()
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    if comparison_data.get_outcomes(i, j):
                        already_compared.add((i, j))

            # Compute |s_i - s_j| for all UNCOMPARED pairs
            pair_uncertainties = []
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    if (i, j) not in already_compared:
                        diff = abs(strengths[i] - strengths[j])
                        pair_uncertainties.append(((i, j), diff))

            # If we've compared everything, allow reuse (shouldn't happen with 500 comps)
            if not pair_uncertainties:
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        diff = abs(strengths[i] - strengths[j])
                        pair_uncertainties.append(((i, j), diff))

            # Sort by descending difference (largest = lowest uncertainty)
            pair_uncertainties.sort(key=lambda x: x[1], reverse=True)

            # Select top batch_size pairs with LARGEST differences
            self._current_batch_pairs = [
                pair for pair, _ in pair_uncertainties[:self.batch_size]
            ]
            self._batch_index = 0

        except Exception as e:
            # If fitting fails (e.g., too few comparisons), fall back to random
            print(f"Warning: Model fitting failed ({e}). Using random sampling.")
            self._current_batch_pairs = []
            self._batch_index = 0

    def _random_pair(self, n_items: int) -> tuple[int, int]:
        """Select a random pair."""
        i, j = np.random.choice(n_items, size=2, replace=False)
        return tuple(sorted([i, j]))

    def __repr__(self) -> str:
        return (
            f"AntiActiveSampler("
            f"batch_size={self.batch_size}, "
            f"burn_in_fraction={self.burn_in_fraction})"
        )
