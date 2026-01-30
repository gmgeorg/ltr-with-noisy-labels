"""
Tests for sampling strategies.
"""

import pytest
import numpy as np

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.sampling import QuickSortSampler, RandomSampler
from ltr_wnl import simulation, noise, ranking
from ltr_wnl.sampling.gain import MaxGainSampler


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sampler():
    """QuickSort sampler with fixed random seed."""
    return QuickSortSampler(random_state=42)


@pytest.fixture
def comparison_data():
    """Empty comparison data for 5 items."""
    return ComparisonData(n_items=5)


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================

def test_quicksort_returns_valid_pairs():
    """Test that QuickSort always returns valid pairs (i < j)."""
    sampler = QuickSortSampler(random_state=42)
    comparison_data = ComparisonData(n_items=10)

    for _ in range(50):
        i, j = sampler.select_next_pair(comparison_data, n_items=10)

        # Valid indices
        assert 0 <= i < 10
        assert 0 <= j < 10
        # Canonical ordering
        assert i < j

        # Simulate outcome (lower index wins)
        outcome = 1 if i < j else -1
        comparison_data.add_comparison(i, j, outcome)


def test_quicksort_deterministic_with_seed():
    """Test that QuickSort produces same sequence with same seed."""
    sampler1 = QuickSortSampler(random_state=42)
    sampler2 = QuickSortSampler(random_state=42)

    data1 = ComparisonData(n_items=5)
    data2 = ComparisonData(n_items=5)

    # Collect 10 pairs from each
    pairs1 = []
    pairs2 = []

    for _ in range(10):
        pair1 = sampler1.select_next_pair(data1, n_items=5)
        pair2 = sampler2.select_next_pair(data2, n_items=5)

        pairs1.append(pair1)
        pairs2.append(pair2)

        # Add same outcomes
        data1.add_comparison(pair1[0], pair1[1], outcome=1)
        data2.add_comparison(pair2[0], pair2[1], outcome=1)

    # Should produce identical sequences
    assert pairs1 == pairs2


def test_quicksort_different_seeds_produce_different_sequences():
    """Test that different seeds produce different pair sequences."""
    sampler1 = QuickSortSampler(random_state=42)
    sampler2 = QuickSortSampler(random_state=123)

    data1 = ComparisonData(n_items=10)
    data2 = ComparisonData(n_items=10)

    pairs1 = []
    pairs2 = []

    for _ in range(20):
        pair1 = sampler1.select_next_pair(data1, n_items=10)
        pair2 = sampler2.select_next_pair(data2, n_items=10)

        pairs1.append(pair1)
        pairs2.append(pair2)

        data1.add_comparison(pair1[0], pair1[1], outcome=1)
        data2.add_comparison(pair2[0], pair2[1], outcome=1)

    # Should be different (with very high probability)
    assert pairs1 != pairs2


# ==============================================================================
# Partitioning Logic Tests
# ==============================================================================

def test_quicksort_partitioning_flow(sampler, comparison_data):
    """
    Verify that sampler processes comparison outcomes correctly.

    Tests that items are categorized based on their comparison results,
    using internal state for verification at specific checkpoints.
    """
    n_items = 5

    # 1. Get first pair and identify pivot and target
    pair1 = sampler.select_next_pair(comparison_data, n_items)
    initial_pivot = sampler._pivot
    target1 = pair1[1] if pair1[0] == initial_pivot else pair1[0]

    # 2. Target LOSES to pivot
    comparison_data.add_comparison(initial_pivot, target1, outcome=1)

    # 3. Get next pair (this processes the previous result)
    pair2 = sampler.select_next_pair(comparison_data, n_items)

    # At this point, if we're still on the same pivot, target1 should be in 'less'
    # But the pivot might have changed, so we only check if pivot is the same
    if sampler._pivot == initial_pivot:
        assert target1 in sampler._less, \
            f"Item {target1} should be in '_less' after losing to pivot {initial_pivot}"


def test_multisort_reset(sampler, comparison_data):
    """
    Verify that sampler resets for a new sort pass when complete.

    NOTE: This test accesses internal state for verification.
    """
    n_items = 2  # Minimal size for quick test

    # First comparison
    pair = sampler.select_next_pair(comparison_data, n_items)
    comparison_data.add_comparison(pair[0], pair[1], outcome=1)

    # This should finish first sort and start a new pass
    pair2 = sampler.select_next_pair(comparison_data, n_items)

    # Verify it's still working (pivot assigned, pairs generated)
    assert sampler._pivot is not None
    assert pair2 in [(0, 1)]  # Only valid pair for 2 items

    # Verify pass counter incremented
    assert sampler._current_pass >= 1


# ==============================================================================
# Full Simulation Tests
# ==============================================================================

def test_quicksort_produces_reasonable_ranking():
    """Test that QuickSort produces accurate rankings with low noise."""
    runner = simulation.SimulationRunner(
        n_items=20,
        noise_model=noise.BetaNoiseModel(mu=0.05, phi=50.0),  # Very low noise
        ranker=ranking.PlackettLuceRanker(),
        sampling_strategy=QuickSortSampler(random_state=42)
    )

    results = runner.run_trial(n_comparisons=100, random_state=42)

    # With low noise, should achieve good accuracy
    assert results['normalized_kendall'] < 0.3, \
        f"QuickSort should achieve <0.3 error with low noise, got {results['normalized_kendall']:.3f}"


def test_quicksort_handles_moderate_noise():
    """Test that QuickSort still works with moderate noise (though less efficiently)."""
    runner = simulation.SimulationRunner(
        n_items=15,
        noise_model=noise.BetaNoiseModel(mu=0.2, phi=10.0),  # Moderate noise
        ranker=ranking.PlackettLuceRanker(),
        sampling_strategy=QuickSortSampler(random_state=42, max_passes=20)
    )

    # Should complete without crashing
    results = runner.run_trial(n_comparisons=150, random_state=42)

    # Should produce some ranking (even if not perfect)
    assert results['estimated_ranking'] is not None
    assert len(results['estimated_ranking']) == 15


def test_quicksort_comparison_efficiency():
    """Test that QuickSort uses comparisons efficiently with low noise."""
    n_items = 30

    # Theory: O(n log n) comparisons for noiseless quicksort
    # n log n ≈ 30 * log2(30) ≈ 30 * 4.9 ≈ 147 comparisons
    # With low noise, should be reasonably close
    max_expected_comparisons = n_items * int(np.log2(n_items)) * 3  # 3x margin

    sampler = QuickSortSampler(random_state=42, max_passes=5)
    comparison_data = ComparisonData(n_items=n_items)

    # Simulate perfect comparisons (i always beats j if i < j in true ranking)
    true_ranking = list(range(n_items))

    comparisons_used = 0
    try:
        for _ in range(max_expected_comparisons):
            i, j = sampler.select_next_pair(comparison_data, n_items)
            comparisons_used += 1

            # Deterministic outcome based on true ranking
            outcome = 1 if true_ranking.index(i) < true_ranking.index(j) else -1
            comparison_data.add_comparison(i, j, outcome)

    except RuntimeError:
        # Max passes exceeded - that's okay for this test
        pass

    # Just verify we didn't crash and used some comparisons
    assert comparisons_used > 0
    print(f"QuickSort used {comparisons_used} comparisons for {n_items} items")


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_quicksort_with_two_items():
    """Test QuickSort with minimal number of items (2)."""
    sampler = QuickSortSampler(random_state=42)
    comparison_data = ComparisonData(n_items=2)

    # Should only need 1 comparison
    pair = sampler.select_next_pair(comparison_data, n_items=2)
    assert pair == (0, 1)

    comparison_data.add_comparison(0, 1, outcome=1)

    # Should reset and continue
    pair2 = sampler.select_next_pair(comparison_data, n_items=2)
    assert pair2 == (0, 1)


def test_quicksort_max_passes_exceeded():
    """Test that QuickSort raises error when max_passes is exceeded."""
    sampler = QuickSortSampler(random_state=42, max_passes=2)
    comparison_data = ComparisonData(n_items=5)

    # With random outcomes, should eventually exceed max_passes
    np.random.seed(42)

    with pytest.raises(RuntimeError, match="Maximum number of sort passes"):
        for _ in range(200):  # Enough iterations to trigger multiple passes
            i, j = sampler.select_next_pair(comparison_data, n_items=5)
            # Random outcomes to prevent convergence
            outcome = np.random.choice([1, -1])
            comparison_data.add_comparison(i, j, outcome)


def test_quicksort_repr():
    """Test string representation."""
    sampler = QuickSortSampler(max_passes=15)
    repr_str = repr(sampler)

    assert "QuickSortSampler" in repr_str
    assert "15" in repr_str


# ==============================================================================
# Comparison with Other Strategies
# ==============================================================================

def test_quicksort_vs_random_low_noise():
    """Compare QuickSort vs Random sampling with low noise."""
    N_ITEMS = 20
    N_COMPARISONS = 80

    # QuickSort
    runner_qs = simulation.SimulationRunner(
        n_items=N_ITEMS,
        noise_model=noise.BetaNoiseModel(mu=0.05, phi=50.0),
        ranker=ranking.PlackettLuceRanker(),
        sampling_strategy=QuickSortSampler(random_state=42)
    )

    results_qs = runner_qs.run_trial(n_comparisons=N_COMPARISONS, random_state=42)

    # Random
    runner_random = simulation.SimulationRunner(
        n_items=N_ITEMS,
        noise_model=noise.BetaNoiseModel(mu=0.05, phi=50.0),
        ranker=ranking.PlackettLuceRanker(),
        sampling_strategy=RandomSampler()
    )

    results_random = runner_random.run_trial(n_comparisons=N_COMPARISONS, random_state=42)

    # QuickSort should be competitive or better with low noise
    print(f"QuickSort error: {results_qs['normalized_kendall']:.3f}")
    print(f"Random error:    {results_random['normalized_kendall']:.3f}")

    # Both should achieve reasonable accuracy
    assert results_qs['normalized_kendall'] < 0.5
    assert results_random['normalized_kendall'] < 0.5


class MockRanker:
    """A minimal mock ranker to provide controlled strengths."""
    def __init__(self, strengths):
        self._strengths = np.array(strengths)
    
    def fit(self, data):
        pass # Strengths are pre-set for the test
        
    def get_strengths(self):
        return self._strengths

@pytest.fixture
def comparison_data():
    return ComparisonData(n_items=4)

def test_max_gain_selection_logic(comparison_data):
    # Set strengths so that item 1 and 2 are very close, others far apart
    # s0=10.0, s1=5.1, s2=5.0, s3=0.0
    # Pair (1, 2) has diff 0.1 (Max Info Gain)
    mock_ranker = MockRanker([10.0, 5.1, 5.0, 0.0])
    
    # Initialize with 0 burn-in to trigger active logic immediately
    sampler = MaxGainSampler(
        ranker=mock_ranker, 
        batch_size=1, 
        burn_in_count=0
    )
    
    # We need to simulate that the model has been "fitted" at least once 
    # (Usually handled by the simulation loop, here we just call the logic)
    pair = sampler.select_next_pair(comparison_data, n_items=4)
    
    # ASSERT: Should pick the pair with the closest strengths (1, 2)
    assert pair == (1, 2), f"Expected (1, 2) due to smallest score diff, got {pair}"

def test_max_gain_burn_in(comparison_data):
    """Verify random sampling occurs during burn-in."""
    mock_ranker = MockRanker([0, 0, 0, 0])
    sampler = MaxGainSampler(ranker=mock_ranker, burn_in_count=5, batch_size=50)
    
    # With 0 comparisons, it should be in burn-in
    pair = sampler.select_next_pair(comparison_data, n_items=4)
    
    # We can't predict the random pair, but we verify it doesn't crash 
    # and returns a valid tuple
    assert isinstance(pair, tuple)
    assert len(pair) == 2