"""
Tests for noise models and comparison generation.

Tests the core stochastic components that generate noisy pairwise comparisons:
- BetaNoiseModel parameter validation and sampling
- ComparisonGenerator BTL probability calculations and outcome generation
"""

import numpy as np
import pytest

from ltr_wnl.noise.beta_noise import BetaNoiseModel
from ltr_wnl.comparisons.generators import ComparisonGenerator


class TestBetaNoiseModel:
    """Test suite for BetaNoiseModel."""

    def test_initialization_valid_params(self):
        """Test BetaNoiseModel with valid parameters."""
        noise = BetaNoiseModel(mu=0.3, phi=10.0)
        assert noise.mu == 0.3
        assert noise.phi == 10.0

    def test_initialization_invalid_mu_too_low(self):
        """Test BetaNoiseModel rejects mu <= 0."""
        with pytest.raises(ValueError, match="mu must be in range"):
            BetaNoiseModel(mu=0.0, phi=10.0)

    def test_initialization_invalid_mu_too_high(self):
        """Test BetaNoiseModel rejects mu >= 1."""
        with pytest.raises(ValueError, match="mu must be in range"):
            BetaNoiseModel(mu=1.0, phi=10.0)

    def test_initialization_invalid_phi(self):
        """Test BetaNoiseModel rejects phi <= 0."""
        with pytest.raises(ValueError, match="phi must be positive"):
            BetaNoiseModel(mu=0.3, phi=0.0)

        with pytest.raises(ValueError, match="phi must be positive"):
            BetaNoiseModel(mu=0.3, phi=-5.0)

    def test_beta_params_computation(self):
        """Test correct computation of Beta(α, β) from (μ, φ)."""
        noise = BetaNoiseModel(mu=0.3, phi=10.0)
        alpha, beta = noise.get_params()

        # α = μ × φ = 0.3 × 10 = 3.0
        # β = (1 - μ) × φ = 0.7 × 10 = 7.0
        assert alpha == pytest.approx(3.0)
        assert beta == pytest.approx(7.0)

    def test_sample_error_rate_in_valid_range(self):
        """Test that sampled error rates are in [0, 1]."""
        noise = BetaNoiseModel(mu=0.3, phi=10.0)

        np.random.seed(42)
        for _ in range(100):
            error_rate = noise.sample_error_rate()
            assert 0 <= error_rate <= 1

    def test_sample_error_rate_mean_convergence(self):
        """Test that sampled error rates converge to mu."""
        noise = BetaNoiseModel(mu=0.3, phi=50.0)  # High phi for low variance

        np.random.seed(42)
        samples = [noise.sample_error_rate() for _ in range(10000)]
        sample_mean = np.mean(samples)

        # With phi=50, variance is low, so sample mean should be close to mu
        assert sample_mean == pytest.approx(0.3, abs=0.02)

    def test_high_phi_low_variance(self):
        """Test that high phi produces low variance in error rates."""
        noise_high_phi = BetaNoiseModel(mu=0.3, phi=100.0)

        np.random.seed(42)
        samples = [noise_high_phi.sample_error_rate() for _ in range(1000)]
        variance = np.var(samples)

        # High concentration should give low variance
        assert variance < 0.005

    def test_low_phi_high_variance(self):
        """Test that low phi produces high variance in error rates."""
        noise_low_phi = BetaNoiseModel(mu=0.3, phi=2.0)

        np.random.seed(42)
        samples = [noise_low_phi.sample_error_rate() for _ in range(1000)]
        variance = np.var(samples)

        # Low concentration should give high variance
        assert variance > 0.02

    def test_repr(self):
        """Test string representation."""
        noise = BetaNoiseModel(mu=0.3, phi=10.0)
        assert repr(noise) == "BetaNoiseModel(mu=0.3, phi=10.0)"


class TestComparisonGenerator:
    """Test suite for ComparisonGenerator."""

    def test_initialization(self):
        """Test ComparisonGenerator initialization."""
        strengths = np.array([1.0, 2.0, 3.0])
        noise = BetaNoiseModel(mu=0.1, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        assert generator.n_items == 3
        assert np.array_equal(generator.strengths, strengths)
        assert generator.noise_model is noise

    def test_btl_probability_equal_strengths(self):
        """Test BTL probability when items have equal strength."""
        strengths = np.array([2.0, 2.0, 2.0])
        noise = BetaNoiseModel(mu=0.1, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        # P(i > j) should be 0.5 when s_i = s_j
        prob = generator.btl_probability(0, 1)
        assert prob == pytest.approx(0.5)

    def test_btl_probability_stronger_item(self):
        """Test BTL probability when one item is much stronger."""
        strengths = np.array([0.0, 5.0])  # Item 1 is much stronger
        noise = BetaNoiseModel(mu=0.1, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        # P(0 > 1) should be very low
        prob_0_wins = generator.btl_probability(0, 1)
        assert prob_0_wins < 0.1

        # P(1 > 0) should be very high
        prob_1_wins = generator.btl_probability(1, 0)
        assert prob_1_wins > 0.9

    def test_btl_probability_computation(self):
        """Test exact BTL probability computation."""
        strengths = np.array([0.0, 1.0])
        noise = BetaNoiseModel(mu=0.1, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        # P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j))
        # P(0 > 1) = exp(0) / (exp(0) + exp(1)) = 1 / (1 + e) ≈ 0.2689
        prob = generator.btl_probability(0, 1)
        expected = 1.0 / (1.0 + np.exp(1.0))
        assert prob == pytest.approx(expected)

    def test_btl_probability_symmetry(self):
        """Test that BTL probabilities sum to 1."""
        strengths = np.array([1.5, 2.3])
        noise = BetaNoiseModel(mu=0.1, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        prob_i_wins = generator.btl_probability(0, 1)
        prob_j_wins = generator.btl_probability(1, 0)

        # Should sum to 1
        assert prob_i_wins + prob_j_wins == pytest.approx(1.0)

    def test_generate_outcome_returns_valid_values(self):
        """Test that generated outcomes are always ±1."""
        strengths = np.array([1.0, 2.0, 3.0])
        noise = BetaNoiseModel(mu=0.2, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        np.random.seed(42)
        for i in range(3):
            for j in range(3):
                if i != j:
                    outcome = generator.generate_outcome(i, j)
                    assert outcome in {1, -1}

    def test_generate_outcome_no_noise(self):
        """Test outcomes without noise follow BTL probabilities."""
        # Create a scenario with very low noise
        strengths = np.array([0.0, 5.0])  # Item 1 is much stronger
        noise = BetaNoiseModel(mu=0.01, phi=1000.0)  # Very low noise
        generator = ComparisonGenerator(strengths, noise)

        np.random.seed(42)
        outcomes = [generator.generate_outcome(0, 1) for _ in range(100)]

        # Item 1 should win almost all comparisons (outcome = -1)
        wins_by_item_1 = sum(1 for o in outcomes if o == -1)
        assert wins_by_item_1 > 85  # Should be >90% with low noise

    def test_generate_outcome_high_noise(self):
        """Test that high noise increases randomness."""
        # Even with different strengths, high noise makes outcomes more random
        strengths = np.array([0.0, 2.0])
        noise = BetaNoiseModel(mu=0.45, phi=10.0)  # High noise
        generator = ComparisonGenerator(strengths, noise)

        np.random.seed(42)
        outcomes = [generator.generate_outcome(0, 1) for _ in range(1000)]

        # With high noise, weaker item should win more often
        wins_by_item_0 = sum(1 for o in outcomes if o == 1)

        # Should be more than 20% (much higher than with no noise)
        assert wins_by_item_0 > 200

    def test_generate_outcome_reproducibility(self):
        """Test that outcomes are reproducible with random seed."""
        strengths = np.array([1.0, 2.0, 3.0])
        noise = BetaNoiseModel(mu=0.2, phi=10.0)

        # First run
        generator1 = ComparisonGenerator(strengths, noise)
        np.random.seed(42)
        outcomes1 = [generator1.generate_outcome(0, 1) for _ in range(10)]

        # Second run with same seed
        generator2 = ComparisonGenerator(strengths, noise)
        np.random.seed(42)
        outcomes2 = [generator2.generate_outcome(0, 1) for _ in range(10)]

        assert outcomes1 == outcomes2

    def test_generate_outcome_statistical_distribution(self):
        """Test that outcome distribution matches expected BTL + noise."""
        strengths = np.array([0.0, 1.0])
        noise = BetaNoiseModel(mu=0.2, phi=100.0)  # Low variance noise
        generator = ComparisonGenerator(strengths, noise)

        np.random.seed(42)
        n_trials = 10000
        outcomes = [generator.generate_outcome(0, 1) for _ in range(n_trials)]
        wins_by_0 = sum(1 for o in outcomes if o == 1)
        empirical_prob_0_wins = wins_by_0 / n_trials

        # P(0 > 1) ≈ 0.2689 (BTL)
        # With 20% noise: P_observed ≈ 0.2689 * 0.8 + (1 - 0.2689) * 0.2
        # ≈ 0.2151 + 0.1462 ≈ 0.3613
        btl_prob = generator.btl_probability(0, 1)
        expected_prob = btl_prob * (1 - 0.2) + (1 - btl_prob) * 0.2

        assert empirical_prob_0_wins == pytest.approx(expected_prob, abs=0.02)

    def test_repr(self):
        """Test string representation."""
        strengths = np.array([1.0, 2.0, 3.0])
        noise = BetaNoiseModel(mu=0.3, phi=10.0)
        generator = ComparisonGenerator(strengths, noise)

        repr_str = repr(generator)
        assert "ComparisonGenerator" in repr_str
        assert "n_items=3" in repr_str
        assert "BetaNoiseModel" in repr_str
