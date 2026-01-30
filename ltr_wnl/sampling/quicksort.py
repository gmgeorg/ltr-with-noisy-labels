"""
Quicksort-based adaptive sampling strategy.

Implements the 'Multisort' algorithm from:
Ailon, N. (2012). "An Active Learning Algorithm for Ranking from Pairwise Preferences
with an Almost Optimal Query Complexity." https://arxiv.org/pdf/1502.05556

WARNING: This algorithm assumes noiseless or low-noise comparisons. Performance
may degrade significantly with high error rates.
"""

import numpy as np

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.sampling.strategy_base import SamplerStrategy


class QuickSortSampler(SamplerStrategy):
    """
    Adaptive sampling based on quicksort partitioning.

    Implements the 'Multisort' algorithm which repeatedly performs quicksort-like
    partitioning passes over items. Each comparison result is used to partition
    items into those that beat/lose to a randomly selected pivot.

    The algorithm works as follows:
    1. Select a random pivot from current partition
    2. Compare all other items against the pivot
    3. Partition items into 'less than' and 'greater than' groups
    4. Recursively process partitions
    5. When complete, restart for another pass (Multisort)

    Theoretical comparison complexity: O(n log n) for noiseless comparisons.

    WARNING: This assumes LOW NOISE. With noisy comparisons (e.g., Beta(μ=0.3)),
    partitioning errors compound through recursion, potentially requiring many
    more comparisons than O(n log n).

    Parameters
    ----------
    random_state : int | None, default=None
        Random seed for reproducible pivot selection
    max_passes : int, default=10
        Maximum number of complete sort passes before stopping

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>> from ltr_wnl.sampling import QuickSortSampler
    >>>
    >>> strategy = QuickSortSampler(random_state=42)
    >>> data = ComparisonData(n_items=10)
    >>> i, j = strategy.select_next_pair(data, n_items=10)
    >>> # Compare items i and j, add result to data
    >>> # Next call will use result to partition items
    """

    def __init__(
        self,
        random_state: int | None = None,
        max_passes: int = 10
    ):
        super().__init__()
        self.rng = np.random.default_rng(random_state)
        self.max_passes = max_passes

        # Internal state
        self._n_items: int | None = None
        self._current_pass: int = 0
        self._stack: list[list[int]] = []  # Stack of partitions to process

        # State for current partition
        self._pivot: int | None = None
        self._to_compare: list[int] = []  # Items remaining to compare vs pivot
        self._less: list[int] = []  # Items that lost to pivot
        self._greater: list[int] = []  # Items that beat pivot
        self._last_pair: tuple[int, int] | None = None  # Last suggested pair

    def select_next_pair(
        self,
        comparison_data: ComparisonData,
        n_items: int
    ) -> tuple[int, int]:
        """
        Select next pair using quicksort partitioning logic.

        Processes the result of the previous comparison (if any), then selects
        the next item to compare against the current pivot.

        Parameters
        ----------
        comparison_data : ComparisonData
            History of previous comparisons
        n_items : int
            Total number of items

        Returns
        -------
        i, j : tuple[int, int]
            Next pair to compare (i < j)

        Raises
        ------
        RuntimeError
            If maximum number of passes is exceeded
        """
        if self._n_items is None:
            self._n_items = n_items
            self._reset_sort()

        # 1. Process the result of the LAST comparison we suggested
        if self._last_pair is not None:
            self._update_state_from_data(comparison_data)

        # 2. If we've finished the current partition, move to the next sub-problem
        while not self._to_compare:
            # Current pivot is done: push sub-partitions to the stack
            if self._pivot is not None:
                # Items that lost to pivot go to one side, winners to the other
                if len(self._less) > 1:
                    self._stack.append(self._less)
                if len(self._greater) > 1:
                    self._stack.append(self._greater)

            if not self._stack:
                # Full sort pass complete! Restart for Multisort.
                self._reset_sort()

            items = self._stack.pop()
            self._pivot = self.rng.choice(items)
            self._to_compare = [i for i in items if i != self._pivot]
            self._less = []
            self._greater = []

        # 3. Pick next item to compare against pivot
        target = self._to_compare.pop()
        self._last_pair = (self._pivot, target)

        return tuple(sorted(self._last_pair))

    def _update_state_from_data(self, comparison_data: ComparisonData) -> None:
        """
        Process the result of the last comparison and update partition state.

        Uses the most recent comparison outcome to classify the target item
        as either 'less than' or 'greater than' the current pivot.

        Parameters
        ----------
        comparison_data : ComparisonData
            Comparison history containing the result of the last suggested pair

        Notes
        -----
        ComparisonData.get_outcomes(p, target) always returns outcomes from
        p's perspective:
        - outcome = 1 means p won (target lost to pivot)
        - outcome = -1 means target won (target beat pivot)
        """
        p, target = self._last_pair
        outcomes = comparison_data.get_outcomes(p, target)

        if outcomes:
            # get_outcomes returns 1 if p (pivot) won, -1 if target won
            if outcomes[-1] == 1:
                self._less.append(target)  # target lost to pivot
            else:
                self._greater.append(target)  # target beat pivot
        self._last_pair = None

    def _reset_sort(self) -> None:
        """
        Initialize a full sorting pass across all items.

        Increments the pass counter and resets the partition stack to
        contain all items. Called when a complete pass finishes.

        Raises
        ------
        RuntimeError
            If maximum number of passes is exceeded
        """
        self._current_pass += 1

        if self._current_pass > self.max_passes:
            raise RuntimeError(
                f"Maximum number of sort passes ({self.max_passes}) exceeded. "
                "This may indicate high noise levels incompatible with quicksort."
            )

        self._stack = [list(range(self._n_items))]
        self._pivot = None
        self._to_compare = []
        self._last_pair = None

    def __repr__(self) -> str:
        return f"QuickSortSampler(max_passes={self.max_passes})"