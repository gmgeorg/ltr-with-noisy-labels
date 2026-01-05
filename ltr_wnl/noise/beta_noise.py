"""
Beta-distributed noise model for pairwise comparisons.
"""

import numpy as np
import scipy.stats as stats

from ltr_wnl.noise.noise_base import NoiseModel


class BetaNoiseModel(NoiseModel):
    """
    Beta-distributed comparison noise model.

    The error probability for each comparison is sampled from a
    Beta(α, β) distribution, where:
    - α = μ × φ
    - β = (1 - μ) × φ

    Parameters
    ----------
    mu : float
        Mean error rate, must be in range (0, 1)
    phi : float
        Concentration parameter, must be positive
        - Low φ: high variance/uncertainty in error rates
        - High φ: low variance/uncertainty in error rates

    Attributes
    ----------
    mu : float
        Mean error rate
    phi : float
        Concentration parameter

    Examples
    --------
    >>> # High uncertainty noise (low concentration)
    >>> noise_high_uncertainty = BetaNoiseModel(mu=0.3, phi=2.0)
    >>>
    >>> # Low uncertainty noise (high concentration)
    >>> noise_low_uncertainty = BetaNoiseModel(mu=0.3, phi=50.0)
    """

    def __init__(self, mu: float, phi: float):
        if not (0 < mu < 1):
            raise ValueError(f"mu must be in range (0, 1), got {mu}")
        if phi <= 0:
            raise ValueError(f"phi must be positive, got {phi}")

        self.mu = mu
        self.phi = phi
        self._alpha, self._beta = self._compute_beta_params()

    def _compute_beta_params(self) -> tuple[float, float]:
        """
        Convert moment parameters (μ, φ) to shape parameters (α, β).

        Returns
        -------
        alpha : float
            Shape parameter α = μ × φ
        beta : float
            Shape parameter β = (1 - μ) × φ
        """
        alpha = self.mu * self.phi
        beta = (1 - self.mu) * self.phi
        return alpha, beta

    def sample_error_rate(
        self,
        i: int | None = None,
        j: int | None = None
    ) -> float:
        """
        Sample comparison error rate from Beta(α, β).

        Parameters
        ----------
        i : int, optional
            Index of first item (unused in this model)
        j : int, optional
            Index of second item (unused in this model)

        Returns
        -------
        error_rate : float
            Sampled error probability from Beta distribution
        """
        return stats.beta.rvs(a=self._alpha, b=self._beta)

    def get_params(self) -> tuple[float, float]:
        """
        Get Beta distribution shape parameters.

        Returns
        -------
        alpha : float
            Shape parameter α
        beta : float
            Shape parameter β
        """
        return self._alpha, self._beta

    def __repr__(self) -> str:
        return f"BetaNoiseModel(mu={self.mu}, phi={self.phi})"
