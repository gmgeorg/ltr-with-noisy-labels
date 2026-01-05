"""
Generate noisy pairwise comparison outcomes.
"""

import numpy as np

from ltr_wnl.noise.noise_base import NoiseModel


class ComparisonGenerator:
    """
    Generate noisy pairwise comparison outcomes.

    Uses the Bradley-Terry-Luce (BTL) model combined with a noise model
    to generate realistic comparison outcomes.

    Parameters
    ----------
    strengths : np.ndarray, shape (n_items,)
        Ground truth latent strength scores for items
    noise_model : NoiseModel
        Noise model for generating comparison errors

    Attributes
    ----------
    strengths : np.ndarray
        Ground truth strengths
    noise_model : NoiseModel
        Noise model
    n_items : int
        Number of items

    Examples
    --------
    >>> from ltr_wnl.noise import BetaNoiseModel
    >>> from ltr_wnl.ground_truth import generate_btl_strengths
    >>>
    >>> strengths = generate_btl_strengths(10, random_state=42)
    >>> noise = BetaNoiseModel(mu=0.3, phi=10.0)
    >>> generator = ComparisonGenerator(strengths, noise)
    >>>
    >>> # Generate comparison between items 0 and 1
    >>> outcome = generator.generate_outcome(0, 1)
    >>> outcome in {1, -1}
    True
    """

    def __init__(self, strengths: np.ndarray, noise_model: NoiseModel):
        self.strengths = strengths
        self.noise_model = noise_model
        self.n_items = len(strengths)

    def btl_probability(self, i: int, j: int) -> float:
        """
        Compute BTL probability P(i > j | strengths).

        Under the Bradley-Terry-Luce model:
        P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j))

        Parameters
        ----------
        i : int
            Index of first item
        j : int
            Index of second item

        Returns
        -------
        prob : float
            Probability that item i is preferred over item j
        """
        s_i = self.strengths[i]
        s_j = self.strengths[j]

        # Use log-sum-exp trick for numerical stability
        # P(i > j) = 1 / (1 + exp(s_j - s_i))
        return 1.0 / (1.0 + np.exp(s_j - s_i))

    def generate_outcome(self, i: int, j: int) -> int:
        """
        Generate noisy comparison outcome for items i vs j.

        Process:
        1. Determine true winner based on BTL probability
        2. Sample error rate from noise model
        3. Flip outcome with probability = error rate

        Parameters
        ----------
        i : int
            Index of first item
        j : int
            Index of second item

        Returns
        -------
        outcome : int
            1 if i wins, -1 if j wins
        """
        # 1. Sample true outcome based on BTL probability
        p_i_wins = self.btl_probability(i, j)
        true_i_wins = np.random.rand() < p_i_wins

        # 2. Sample error probability from noise model
        p_error = self.noise_model.sample_error_rate(i, j)

        # 3. Potentially flip the outcome
        is_error = np.random.rand() < p_error

        # Determine final outcome
        if true_i_wins and not is_error:
            return 1  # i wins correctly
        elif true_i_wins and is_error:
            return -1  # j wins due to error
        elif not true_i_wins and not is_error:
            return -1  # j wins correctly
        else:
            return 1  # i wins due to error

    def __repr__(self) -> str:
        return (
            f"ComparisonGenerator(n_items={self.n_items}, "
            f"noise_model={self.noise_model})"
        )
