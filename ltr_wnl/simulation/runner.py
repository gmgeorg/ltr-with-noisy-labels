"""
Core simulation runner for single trials.
"""

from typing import Any

import numpy as np

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.comparisons.generators import ComparisonGenerator
from ltr_wnl.ground_truth.generators import (
    generate_btl_strengths,
    strengths_to_ranking,
)
from ltr_wnl.metrics.ranking_metrics import (
    kendall_tau_distance,
    normalized_kendall_tau,
)
from ltr_wnl.noise.noise_base import NoiseModel
from ltr_wnl.ranking.ranker_base import Ranker
from ltr_wnl.sampling.strategy_base import SamplerStrategy


class SimulationRunner:
    """
    Run a single simulation trial.

    Orchestrates the full simulation pipeline:
    1. Generate ground truth (if not provided)
    2. Select pairs using sampling strategy
    3. Generate noisy comparisons
    4. Infer ranking using ranker
    5. Evaluate ranking quality

    Parameters
    ----------
    n_items : int
        Number of items to rank
    noise_model : NoiseModel
        Noise model for generating comparison errors
    ranker : Ranker
        Ranking algorithm
    sampling_strategy : SamplerStrategy
        Strategy for selecting which pairs to compare

    Examples
    --------
    >>> from ltr_wnl.noise import BetaNoiseModel
    >>> from ltr_wnl.ranking import BordaCountRanker
    >>> from ltr_wnl.sampling import RandomSampler
    >>>
    >>> runner = SimulationRunner(
    ...     n_items=10,
    ...     noise_model=BetaNoiseModel(mu=0.3, phi=10.0),
    ...     ranker=BordaCountRanker(),
    ...     sampling_strategy=RandomSampler()
    ... )
    >>>
    >>> results = runner.run_trial(n_comparisons=50, random_state=42)
    >>> results['kendall_distance']
    """

    def __init__(
        self,
        n_items: int,
        noise_model: NoiseModel,
        ranker: Ranker,
        sampling_strategy: SamplerStrategy
    ):
        self.n_items = n_items
        self.noise_model = noise_model
        self.ranker = ranker
        self.sampling_strategy = sampling_strategy

    def run_trial(
        self,
        n_comparisons: int,
        ground_truth_strengths: np.ndarray | None = None,
        random_state: int | None = None
    ) -> dict[str, Any]:
        """
        Run a single simulation trial.

        Parameters
        ----------
        n_comparisons : int
            Number of pairwise comparisons to collect
        ground_truth_strengths : np.ndarray, optional
            Pre-specified ground truth strengths. If None, will be
            generated randomly.
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'estimated_ranking': list[int]
            - 'true_ranking': list[int]
            - 'kendall_distance': int
            - 'normalized_kendall': float
            - 'comparison_data': ComparisonData
            - 'ground_truth_strengths': np.ndarray
        """
        if random_state is not None:
            np.random.seed(random_state)

        # 1. Generate or use provided ground truth
        if ground_truth_strengths is None:
            strengths = generate_btl_strengths(
                self.n_items,
                distribution='normal',
                random_state=random_state
            )
        else:
            strengths = ground_truth_strengths

        true_ranking = strengths_to_ranking(strengths)

        # 2. Initialize comparison generator and data storage
        generator = ComparisonGenerator(strengths, self.noise_model)
        comparison_data = ComparisonData(self.n_items)

        # 3. Collect comparisons
        for _ in range(n_comparisons):
            # Select next pair
            i, j = self.sampling_strategy.select_next_pair(
                comparison_data,
                self.n_items
            )

            # Generate noisy outcome
            outcome = generator.generate_outcome(i, j)

            # Store comparison
            comparison_data.add_comparison(i, j, outcome)

        # 4. Infer ranking
        estimated_ranking = self.ranker.fit(comparison_data)

        # 5. Evaluate
        kendall_dist = kendall_tau_distance(estimated_ranking, true_ranking)
        normalized_kendall = normalized_kendall_tau(
            estimated_ranking,
            true_ranking
        )

        rank_correlation = np.corrcoef(estimated_ranking, true_ranking)[0, 1]

        return {
            'estimated_ranking': estimated_ranking,
            'true_ranking': true_ranking,
            'kendall_distance': kendall_dist,
            'normalized_kendall': normalized_kendall,
            'comparison_data': comparison_data,
            'ground_truth_strengths': strengths,
            'rank_correlation': rank_correlation,
        }

    def run_batch_trial(
        self,
        n_comparisons: int,
        batch_size: int,
        ground_truth_strengths: np.ndarray | None = None,
        random_state: int | None = None
    ) -> dict[str, Any]:
        """
        Run trial with batch-based evaluation.

        Collects comparisons in batches and evaluates ranking quality
        after each batch. Useful for comparing active learning strategies.

        Parameters
        ----------
        n_comparisons : int
            Total number of pairwise comparisons to collect
        batch_size : int
            Number of comparisons per batch (evaluate after each batch)
        ground_truth_strengths : np.ndarray, optional
            Pre-specified ground truth strengths. If None, will be
            generated randomly.
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'comparisons_used': list[int] - cumulative comparisons at each checkpoint
            - 'kendall_distances': list[float] - Kendall distance at each checkpoint
            - 'normalized_kendalls': list[float] - normalized Kendall at each checkpoint
            - 'rank_correlations': list[float] - rank correlation at each checkpoint
            - 'final_ranking': list[int] - final estimated ranking
            - 'true_ranking': list[int] - true ranking
            - 'ground_truth_strengths': np.ndarray - ground truth strengths
        """
        if random_state is not None:
            np.random.seed(random_state)

        # 1. Generate or use provided ground truth
        if ground_truth_strengths is None:
            strengths = generate_btl_strengths(
                self.n_items,
                distribution='normal',
                random_state=random_state
            )
        else:
            strengths = ground_truth_strengths

        true_ranking = strengths_to_ranking(strengths)

        # 2. Initialize comparison generator and data storage
        generator = ComparisonGenerator(strengths, self.noise_model)
        comparison_data = ComparisonData(self.n_items)

        # 3. Tracking arrays
        comparisons_used = []
        kendall_distances = []
        normalized_kendalls = []
        rank_correlations = []

        # 4. Collect comparisons in batches
        n_batches = (n_comparisons + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            # Determine batch size (last batch may be smaller)
            current_batch_size = min(
                batch_size,
                n_comparisons - batch_idx * batch_size
            )

            # Collect comparisons for this batch
            for _ in range(current_batch_size):
                # Select next pair
                i, j = self.sampling_strategy.select_next_pair(
                    comparison_data,
                    self.n_items
                )

                # Generate noisy outcome
                outcome = generator.generate_outcome(i, j)

                # Store comparison
                comparison_data.add_comparison(i, j, outcome)

            # Evaluate after this batch
            try:
                estimated_ranking = self.ranker.fit(comparison_data)

                kendall_dist = kendall_tau_distance(estimated_ranking, true_ranking)
                normalized_kendall = normalized_kendall_tau(
                    estimated_ranking,
                    true_ranking
                )
                rank_correlation = np.corrcoef(estimated_ranking, true_ranking)[0, 1]

                # Record metrics
                comparisons_used.append(comparison_data.total_comparisons())
                kendall_distances.append(kendall_dist)
                normalized_kendalls.append(normalized_kendall)
                rank_correlations.append(rank_correlation)

            except Exception as e:
                # If ranking fails (e.g., too few comparisons), record NaN
                print(f"Warning: Ranking failed at batch {batch_idx}: {e}")
                comparisons_used.append(comparison_data.total_comparisons())
                kendall_distances.append(np.nan)
                normalized_kendalls.append(np.nan)
                rank_correlations.append(np.nan)

        # 5. Final ranking
        try:
            final_ranking = self.ranker.fit(comparison_data)
        except Exception:
            final_ranking = list(range(self.n_items))  # Fallback to identity

        return {
            'comparisons_used': comparisons_used,
            'kendall_distances': kendall_distances,
            'normalized_kendalls': normalized_kendalls,
            'rank_correlations': rank_correlations,
            'final_ranking': final_ranking,
            'true_ranking': true_ranking,
            'ground_truth_strengths': strengths,
        }

    def __repr__(self) -> str:
        return (
            f"SimulationRunner(\n"
            f"  n_items={self.n_items},\n"
            f"  noise_model={self.noise_model},\n"
            f"  ranker={self.ranker},\n"
            f"  sampling_strategy={self.sampling_strategy}\n"
            f")"
        )
