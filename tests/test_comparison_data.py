"""
Tests for ComparisonData class.

Tests the critical pair canonicalization and data management logic:
- Pair normalization (always storing i < j)
- Outcome flipping when pairs are reversed
- Win counting and matrix computation
- Data retrieval and aggregation
"""

import numpy as np
import pytest

from ltr_wnl.comparisons.data import ComparisonData


class TestComparisonData:
    """Test suite for ComparisonData."""

    def test_initialization(self):
        """Test ComparisonData initialization."""
        data = ComparisonData(n_items=5)
        assert data.n_items == 5
        assert len(data.comparisons) == 0
        assert data.total_comparisons() == 0

    def test_add_comparison_canonical_order(self):
        """Test adding comparison with i < j (canonical order)."""
        data = ComparisonData(n_items=5)
        data.add_comparison(0, 1, outcome=1)  # Item 0 beat item 1

        # Should be stored as (0, 1) -> [1]
        assert (0, 1) in data.comparisons
        assert data.comparisons[(0, 1)] == [1]
        assert data.total_comparisons() == 1

    def test_add_comparison_reversed_order(self):
        """Test adding comparison with i > j (needs reversal)."""
        data = ComparisonData(n_items=5)
        data.add_comparison(1, 0, outcome=1)  # Item 1 beat item 0

        # Should be stored as (0, 1) -> [-1] (pair reversed, outcome flipped)
        assert (0, 1) in data.comparisons
        assert data.comparisons[(0, 1)] == [-1]
        assert (1, 0) not in data.comparisons

    def test_add_comparison_multiple_outcomes(self):
        """Test adding multiple outcomes for the same pair."""
        data = ComparisonData(n_items=5)

        # Add several comparisons for the same pair
        data.add_comparison(0, 1, outcome=1)   # Item 0 wins
        data.add_comparison(0, 1, outcome=-1)  # Item 1 wins
        data.add_comparison(0, 1, outcome=1)   # Item 0 wins

        assert data.comparisons[(0, 1)] == [1, -1, 1]
        assert data.total_comparisons() == 3

    def test_add_comparison_reversed_pairs_same_storage(self):
        """Test that (i,j) and (j,i) are stored in the same location."""
        data = ComparisonData(n_items=5)

        data.add_comparison(0, 1, outcome=1)   # 0 beats 1
        data.add_comparison(1, 0, outcome=1)   # 1 beats 0

        # Both should be stored as (0, 1) with flipped outcomes
        assert (0, 1) in data.comparisons
        assert data.comparisons[(0, 1)] == [1, -1]  # First: 0 wins, Second: 1 wins
        assert (1, 0) not in data.comparisons

    def test_add_comparison_invalid_outcome(self):
        """Test that invalid outcomes are rejected."""
        data = ComparisonData(n_items=5)

        with pytest.raises(ValueError, match="outcome must be 1 or -1"):
            data.add_comparison(0, 1, outcome=0)

        with pytest.raises(ValueError, match="outcome must be 1 or -1"):
            data.add_comparison(0, 1, outcome=2)

    def test_get_outcomes_canonical_order(self):
        """Test retrieving outcomes for pair in canonical order."""
        data = ComparisonData(n_items=5)
        data.add_comparison(0, 1, outcome=1)
        data.add_comparison(0, 1, outcome=-1)

        outcomes = data.get_outcomes(0, 1)
        assert outcomes == [1, -1]

    def test_get_outcomes_reversed_order(self):
        """Test retrieving outcomes for reversed pair (flips outcomes)."""
        data = ComparisonData(n_items=5)
        data.add_comparison(0, 1, outcome=1)   # 0 beats 1
        data.add_comparison(0, 1, outcome=-1)  # 1 beats 0

        # When querying (1, 0), outcomes should be flipped
        outcomes = data.get_outcomes(1, 0)
        assert outcomes == [-1, 1]  # Flipped from [1, -1]

    def test_get_outcomes_no_comparisons(self):
        """Test retrieving outcomes for pair with no comparisons."""
        data = ComparisonData(n_items=5)
        outcomes = data.get_outcomes(0, 1)
        assert outcomes == []

    def test_get_outcomes_symmetry(self):
        """Test that get_outcomes(i,j) and get_outcomes(j,i) are consistent."""
        data = ComparisonData(n_items=5)
        data.add_comparison(2, 3, outcome=1)   # 2 beats 3
        data.add_comparison(3, 2, outcome=-1)  # 2 beats 3 (reversed pair)

        outcomes_2_3 = data.get_outcomes(2, 3)
        outcomes_3_2 = data.get_outcomes(3, 2)

        # Should be negations of each other
        assert outcomes_2_3 == [1, 1]
        assert outcomes_3_2 == [-1, -1]

    def test_get_win_count_canonical_order(self):
        """Test win count for canonical pair order."""
        data = ComparisonData(n_items=5)
        data.add_comparison(0, 1, outcome=1)   # 0 wins
        data.add_comparison(0, 1, outcome=1)   # 0 wins
        data.add_comparison(0, 1, outcome=-1)  # 1 wins

        assert data.get_win_count(0, 1) == 2  # 0 won twice
        assert data.get_win_count(1, 0) == 1  # 1 won once

    def test_get_win_count_reversed_order(self):
        """Test win count for reversed pair order."""
        data = ComparisonData(n_items=5)
        data.add_comparison(1, 0, outcome=1)   # 1 beats 0
        data.add_comparison(0, 1, outcome=1)   # 0 beats 1

        # Both ways should work correctly
        assert data.get_win_count(0, 1) == 1
        assert data.get_win_count(1, 0) == 1

    def test_get_win_count_no_comparisons(self):
        """Test win count for pairs with no comparisons."""
        data = ComparisonData(n_items=5)
        assert data.get_win_count(0, 1) == 0
        assert data.get_win_count(3, 4) == 0

    def test_get_all_pairs(self):
        """Test retrieving all compared pairs."""
        data = ComparisonData(n_items=5)
        data.add_comparison(0, 1, outcome=1)
        data.add_comparison(2, 3, outcome=-1)
        data.add_comparison(1, 4, outcome=1)

        pairs = data.get_all_pairs()
        assert len(pairs) == 3
        assert (0, 1) in pairs
        assert (2, 3) in pairs
        assert (1, 4) in pairs

    def test_get_all_pairs_with_reversed_adds(self):
        """Test that reversed pair additions don't create duplicates."""
        data = ComparisonData(n_items=5)
        data.add_comparison(0, 1, outcome=1)
        data.add_comparison(1, 0, outcome=-1)  # Same pair, reversed

        pairs = data.get_all_pairs()
        assert len(pairs) == 1  # Only one unique pair
        assert (0, 1) in pairs

    def test_get_win_matrix_simple(self):
        """Test win matrix computation for simple case."""
        data = ComparisonData(n_items=3)
        data.add_comparison(0, 1, outcome=1)   # 0 beats 1
        data.add_comparison(0, 2, outcome=1)   # 0 beats 2
        data.add_comparison(1, 2, outcome=-1)  # 2 beats 1

        W = data.get_win_matrix()

        assert W.shape == (3, 3)
        assert W[0, 1] == 1  # 0 beat 1 once
        assert W[0, 2] == 1  # 0 beat 2 once
        assert W[1, 0] == 0  # 1 never beat 0
        assert W[1, 2] == 0  # 1 never beat 2
        assert W[2, 1] == 1  # 2 beat 1 once
        assert W[2, 0] == 0  # 2 never beat 0

    def test_get_win_matrix_multiple_outcomes(self):
        """Test win matrix with multiple comparisons of same pair."""
        data = ComparisonData(n_items=2)
        data.add_comparison(0, 1, outcome=1)   # 0 wins
        data.add_comparison(0, 1, outcome=1)   # 0 wins
        data.add_comparison(0, 1, outcome=-1)  # 1 wins
        data.add_comparison(1, 0, outcome=1)   # 1 wins (reversed pair)

        W = data.get_win_matrix()

        assert W[0, 1] == 2  # 0 beat 1 twice
        assert W[1, 0] == 2  # 1 beat 0 twice

    def test_get_win_matrix_empty(self):
        """Test win matrix for data with no comparisons."""
        data = ComparisonData(n_items=4)
        W = data.get_win_matrix()

        assert W.shape == (4, 4)
        assert np.all(W == 0)

    def test_get_win_matrix_symmetry(self):
        """Test that win matrix captures competitive symmetry."""
        data = ComparisonData(n_items=3)
        data.add_comparison(0, 1, outcome=1)   # 0 beats 1
        data.add_comparison(1, 0, outcome=1)   # 1 beats 0

        W = data.get_win_matrix()

        # Each should have beaten the other once
        assert W[0, 1] == 1
        assert W[1, 0] == 1

        # Total comparisons: W[i,j] + W[j,i] should equal number of comparisons
        assert W[0, 1] + W[1, 0] == 2

    def test_total_comparisons(self):
        """Test total comparison counting."""
        data = ComparisonData(n_items=5)

        assert data.total_comparisons() == 0

        data.add_comparison(0, 1, outcome=1)
        assert data.total_comparisons() == 1

        data.add_comparison(0, 1, outcome=-1)
        assert data.total_comparisons() == 2

        data.add_comparison(2, 3, outcome=1)
        assert data.total_comparisons() == 3

    def test_total_comparisons_with_reversed_pairs(self):
        """Test that total counts reversed pairs correctly."""
        data = ComparisonData(n_items=5)

        data.add_comparison(0, 1, outcome=1)
        data.add_comparison(1, 0, outcome=-1)  # Same pair, reversed

        # Should count both comparisons
        assert data.total_comparisons() == 2

    def test_repr(self):
        """Test string representation."""
        data = ComparisonData(n_items=10)
        data.add_comparison(0, 1, outcome=1)
        data.add_comparison(2, 3, outcome=-1)
        data.add_comparison(2, 3, outcome=1)

        repr_str = repr(data)
        assert "ComparisonData" in repr_str
        assert "n_items=10" in repr_str
        assert "n_pairs=2" in repr_str
        assert "total_comparisons=3" in repr_str

    def test_complex_scenario(self):
        """Test a complex scenario with multiple items and comparisons."""
        data = ComparisonData(n_items=5)

        # Add comparisons in various orders
        comparisons = [
            (0, 1, 1),   # 0 beats 1
            (1, 0, -1),  # 0 beats 1 (reversed)
            (2, 3, 1),   # 2 beats 3
            (3, 2, -1),  # 2 beats 3 (reversed)
            (0, 4, -1),  # 4 beats 0
            (4, 0, 1),   # 4 beats 0 (reversed)
            (1, 2, 1),   # 1 beats 2
        ]

        for i, j, outcome in comparisons:
            data.add_comparison(i, j, outcome)

        # Verify total count
        assert data.total_comparisons() == 7

        # Verify unique pairs
        assert len(data.get_all_pairs()) == 4  # (0,1), (2,3), (0,4), (1,2)

        # Verify specific win counts
        assert data.get_win_count(0, 1) == 2  # 0 beat 1 twice
        assert data.get_win_count(1, 0) == 0  # 1 never beat 0
        assert data.get_win_count(2, 3) == 2  # 2 beat 3 twice
        assert data.get_win_count(4, 0) == 2  # 4 beat 0 twice
        assert data.get_win_count(0, 4) == 0  # 0 never beat 4

        # Verify win matrix
        W = data.get_win_matrix()
        assert W[0, 1] == 2
        assert W[2, 3] == 2
        assert W[4, 0] == 2
        assert W[1, 2] == 1

    def test_outcome_flipping_correctness(self):
        """Test that outcome flipping is mathematically correct."""
        data = ComparisonData(n_items=3)

        # Add: item 0 beats item 1
        data.add_comparison(0, 1, outcome=1)

        # Verify from both perspectives
        assert data.get_outcomes(0, 1) == [1]   # 0 beat 1
        assert data.get_outcomes(1, 0) == [-1]  # 1 lost to 0

        # Add: item 2 beats item 1 (using reversed pair order)
        data.add_comparison(1, 2, outcome=-1)

        # Verify from both perspectives
        assert data.get_outcomes(1, 2) == [-1]  # 1 lost to 2
        assert data.get_outcomes(2, 1) == [1]   # 2 beat 1

        # Verify internal storage is canonical
        assert (0, 1) in data.comparisons
        assert (1, 2) in data.comparisons
        assert (1, 0) not in data.comparisons
        assert (2, 1) not in data.comparisons
