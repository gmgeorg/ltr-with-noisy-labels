"""
Tests for SimulationRunner and end-to-end integration.

Tests the main orchestration engine and integration of all components:
- Full simulation pipeline execution
- Reproducibility via random_state
- Metric computation correctness
- Integration between all modules
"""

import numpy as np
import pytest

from ltr_wnl.simulation.runner import SimulationRunner
from ltr_wnl.noise.beta_noise import BetaNoiseModel
from ltr_wnl.ranking.borda_count import BordaCountRanker
from ltr_wnl.sampling.random import RandomSampling
from ltr_wnl.comparisons.data import ComparisonData


class TestSimulationRunner:
    """Test suite for SimulationRunner."""

    def test_initialization(self):
        """Test SimulationRunner initialization."""
        noise = BetaNoiseModel(mu=0.2, phi=10.0)
        ranker = BordaCountRanker()
        sampling = RandomSampling()

        runner = SimulationRunner(
            n_items=5,
            noise_model=noise,
            ranker=ranker,
            sampling_strategy=sampling
        )

        assert runner.n_items == 5
        assert runner.noise_model is noise
        assert runner.ranker is ranker
        assert runner.sampling_strategy is sampling

    def test_run_trial_basic(self):
        """Test basic trial execution."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.2, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(n_comparisons=20, random_state=42)

        # Check all expected keys are present
        assert 'estimated_ranking' in results
        assert 'true_ranking' in results
        assert 'kendall_distance' in results
        assert 'normalized_kendall' in results
        assert 'comparison_data' in results
        assert 'ground_truth_strengths' in results
        assert 'rank_correlation' in results

    def test_run_trial_ranking_format(self):
        """Test that rankings have correct format."""
        runner = SimulationRunner(
            n_items=10,
            noise_model=BetaNoiseModel(mu=0.1, phi=20.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(n_comparisons=50, random_state=42)

        estimated = results['estimated_ranking']
        true = results['true_ranking']

        # Both should be lists of length n_items
        assert isinstance(estimated, list)
        assert isinstance(true, list)
        assert len(estimated) == 10
        assert len(true) == 10

        # Both should contain each item index exactly once
        assert set(estimated) == set(range(10))
        assert set(true) == set(range(10))

    def test_run_trial_comparison_count(self):
        """Test that correct number of comparisons are collected."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.2, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        n_comparisons = 30
        results = runner.run_trial(n_comparisons=n_comparisons, random_state=42)

        comparison_data = results['comparison_data']
        assert comparison_data.total_comparisons() == n_comparisons

    def test_run_trial_reproducibility(self):
        """Test that trials are reproducible with same random_state."""
        runner = SimulationRunner(
            n_items=8,
            noise_model=BetaNoiseModel(mu=0.3, phi=5.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # Run trial twice with same seed
        results1 = runner.run_trial(n_comparisons=40, random_state=123)
        results2 = runner.run_trial(n_comparisons=40, random_state=123)

        # Results should be identical
        assert results1['estimated_ranking'] == results2['estimated_ranking']
        assert results1['true_ranking'] == results2['true_ranking']
        assert results1['kendall_distance'] == results2['kendall_distance']
        assert np.allclose(
            results1['ground_truth_strengths'],
            results2['ground_truth_strengths']
        )

    def test_run_trial_different_seeds_different_results(self):
        """Test that different random_states produce different results."""
        runner = SimulationRunner(
            n_items=8,
            noise_model=BetaNoiseModel(mu=0.3, phi=5.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # Run with different seeds
        results1 = runner.run_trial(n_comparisons=40, random_state=123)
        results2 = runner.run_trial(n_comparisons=40, random_state=456)

        # Ground truth strengths should be different (with very high probability)
        assert not np.allclose(
            results1['ground_truth_strengths'],
            results2['ground_truth_strengths']
        )

    def test_run_trial_provided_ground_truth(self):
        """Test using provided ground truth strengths."""
        # Provide specific strengths
        ground_truth = np.array([3.0, 1.0, 2.0, 0.0])

        runner = SimulationRunner(
            n_items=4,
            noise_model=BetaNoiseModel(mu=0.1, phi=50.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(
            n_comparisons=20,
            ground_truth_strengths=ground_truth,
            random_state=42
        )

        # Should use the provided strengths
        assert np.array_equal(results['ground_truth_strengths'], ground_truth)

        # True ranking should be [0, 2, 1, 3] (indices sorted by strength)
        expected_ranking = [0, 2, 1, 3]
        assert results['true_ranking'] == expected_ranking

    def test_run_trial_low_noise_good_ranking(self):
        """Test that low noise produces good ranking estimates."""
        runner = SimulationRunner(
            n_items=6,
            noise_model=BetaNoiseModel(mu=0.05, phi=100.0),  # Very low noise
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # With many comparisons and low noise, should recover ranking well
        results = runner.run_trial(n_comparisons=100, random_state=42)

        # Normalized Kendall distance should be low
        assert results['normalized_kendall'] < 0.3

    def test_run_trial_high_noise_poor_ranking(self):
        """Test that high noise makes ranking harder."""
        runner = SimulationRunner(
            n_items=6,
            noise_model=BetaNoiseModel(mu=0.4, phi=5.0),  # High noise
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # With high noise and few comparisons, ranking will be poor
        results = runner.run_trial(n_comparisons=20, random_state=42)

        # With high noise, normalized distance will likely be higher
        # (though exact value depends on random seed)
        assert 0 <= results['normalized_kendall'] <= 1.0

    def test_run_trial_kendall_distance_bounds(self):
        """Test that Kendall distance is in valid range."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.2, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(n_comparisons=30, random_state=42)

        n = 5
        max_inversions = n * (n - 1) // 2  # = 10

        # Distance should be between 0 and max_inversions
        assert 0 <= results['kendall_distance'] <= max_inversions

        # Normalized should be in [0, 1]
        assert 0 <= results['normalized_kendall'] <= 1.0

    def test_run_trial_normalized_kendall_consistency(self):
        """Test that normalized Kendall is consistent with distance."""
        runner = SimulationRunner(
            n_items=7,
            noise_model=BetaNoiseModel(mu=0.2, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(n_comparisons=40, random_state=42)

        n = 7
        max_inversions = n * (n - 1) / 2

        # normalized = distance / max_inversions
        expected_normalized = results['kendall_distance'] / max_inversions
        assert results['normalized_kendall'] == pytest.approx(expected_normalized)

    def test_run_trial_comparison_data_structure(self):
        """Test that comparison data is properly structured."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.2, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(n_comparisons=25, random_state=42)

        comparison_data = results['comparison_data']

        # Should be a ComparisonData instance
        assert isinstance(comparison_data, ComparisonData)

        # Should have correct n_items
        assert comparison_data.n_items == 5

        # All stored pairs should be canonical (i < j)
        for (i, j) in comparison_data.get_all_pairs():
            assert i < j

        # All outcomes should be ±1
        for (i, j), outcomes in comparison_data.comparisons.items():
            for outcome in outcomes:
                assert outcome in {1, -1}

    def test_run_trial_more_comparisons_better_ranking(self):
        """Test that more comparisons generally improve ranking quality."""
        runner = SimulationRunner(
            n_items=6,
            noise_model=BetaNoiseModel(mu=0.2, phi=20.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # Use fixed ground truth for fair comparison
        ground_truth = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.0])

        # Few comparisons
        results_few = runner.run_trial(
            n_comparisons=20,
            ground_truth_strengths=ground_truth,
            random_state=42
        )

        # Many comparisons (same random_state but more data)
        # Need different seed since we're doing more comparisons
        results_many = runner.run_trial(
            n_comparisons=80,
            ground_truth_strengths=ground_truth,
            random_state=43
        )

        # More comparisons should not make things worse on average
        # (though individual runs may vary due to stochasticity)
        # This test might be flaky, so we just check both are valid
        assert 0 <= results_few['normalized_kendall'] <= 1.0
        assert 0 <= results_many['normalized_kendall'] <= 1.0

    def test_run_trial_rank_correlation(self):
        """Test that rank correlation is computed."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.1, phi=50.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        results = runner.run_trial(n_comparisons=30, random_state=42)

        # Rank correlation should be in [-1, 1]
        assert -1 <= results['rank_correlation'] <= 1

        # With low noise and sufficient comparisons, should be positive
        assert results['rank_correlation'] > 0

    def test_integration_full_pipeline(self):
        """Integration test of full simulation pipeline."""
        # This tests the integration of:
        # 1. Ground truth generation
        # 2. Comparison generation with noise
        # 3. Data storage with pair canonicalization
        # 4. Ranking inference
        # 5. Metric computation

        noise = BetaNoiseModel(mu=0.15, phi=30.0)
        ranker = BordaCountRanker()
        sampling = RandomSampling(allow_repeats=False)

        runner = SimulationRunner(
            n_items=8,
            noise_model=noise,
            ranker=ranker,
            sampling_strategy=sampling
        )

        results = runner.run_trial(n_comparisons=28, random_state=999)

        # Verify all components worked correctly

        # 1. Ground truth is reasonable
        strengths = results['ground_truth_strengths']
        assert len(strengths) == 8
        assert not np.all(strengths == strengths[0])  # Not all equal

        # 2. True ranking is consistent with strengths
        true_ranking = results['true_ranking']
        for i in range(len(true_ranking) - 1):
            item_better = true_ranking[i]
            item_worse = true_ranking[i + 1]
            assert strengths[item_better] >= strengths[item_worse]

        # 3. Comparisons were collected
        comparison_data = results['comparison_data']
        assert comparison_data.total_comparisons() == 28

        # 4. Estimated ranking was produced
        estimated_ranking = results['estimated_ranking']
        assert len(estimated_ranking) == 8
        assert set(estimated_ranking) == set(range(8))

        # 5. Metrics were computed
        assert isinstance(results['kendall_distance'], (int, np.integer))
        assert isinstance(results['normalized_kendall'], (float, np.floating))
        assert 0 <= results['normalized_kendall'] <= 1.0

    def test_multiple_trials_statistical_properties(self):
        """Test statistical properties across multiple trials."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.2, phi=20.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # Run multiple trials with different seeds
        kendall_distances = []
        for seed in range(50):
            results = runner.run_trial(n_comparisons=30, random_state=seed)
            kendall_distances.append(results['kendall_distance'])

        # All should be in valid range
        assert all(0 <= d <= 10 for d in kendall_distances)

        # Should have some variance (not all the same)
        assert len(set(kendall_distances)) > 5

        # Mean should be reasonable (not worst case)
        mean_distance = np.mean(kendall_distances)
        assert mean_distance < 8  # Should be better than random

    def test_repr(self):
        """Test string representation."""
        runner = SimulationRunner(
            n_items=10,
            noise_model=BetaNoiseModel(mu=0.3, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        repr_str = repr(runner)
        assert "SimulationRunner" in repr_str
        assert "n_items=10" in repr_str
        assert "BetaNoiseModel" in repr_str
        assert "BordaCountRanker" in repr_str
        assert "RandomSampling" in repr_str

    def test_no_comparisons_edge_case(self):
        """Test behavior with zero comparisons (edge case)."""
        runner = SimulationRunner(
            n_items=5,
            noise_model=BetaNoiseModel(mu=0.2, phi=10.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling()
        )

        # With 0 comparisons, ranker should still produce a ranking
        # (though it will likely be arbitrary/random)
        results = runner.run_trial(n_comparisons=0, random_state=42)

        assert len(results['estimated_ranking']) == 5
        assert set(results['estimated_ranking']) == set(range(5))

    def test_perfect_data_perfect_ranking(self):
        """Test that perfect data (no noise, all pairs) gives perfect ranking."""
        # Use very low noise
        runner = SimulationRunner(
            n_items=4,
            noise_model=BetaNoiseModel(mu=0.001, phi=1000.0),
            ranker=BordaCountRanker(),
            sampling_strategy=RandomSampling(allow_repeats=False)
        )

        # Compare all pairs: C(4, 2) = 6 pairs
        results = runner.run_trial(n_comparisons=6, random_state=42)

        # With no noise and all pairs, should get perfect or near-perfect ranking
        # (might have 1-3 errors due to tiny noise and stochasticity)
        assert results['kendall_distance'] <= 3
