"""
Random state management utilities.
"""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for numpy and random modules.

    This ensures reproducibility across all random number generation
    in the simulation.

    Parameters
    ----------
    seed : int
        Random seed

    Examples
    --------
    >>> set_global_seed(42)
    >>> np.random.rand()
    0.3745401188473625
    """
    np.random.seed(seed)
    random.seed(seed)


class RandomStateContext:
    """
    Context manager for temporary random state.

    This allows you to temporarily set a random seed within a block
    of code, then restore the previous random state afterward.

    Parameters
    ----------
    seed : int, optional
        Random seed to use within the context

    Examples
    --------
    >>> # Generate some random numbers
    >>> np.random.seed(0)
    >>> x1 = np.random.rand()
    >>>
    >>> # Use a different seed temporarily
    >>> with RandomStateContext(42):
    ...     x2 = np.random.rand()
    >>>
    >>> # Back to original stream
    >>> x3 = np.random.rand()
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.np_state = None
        self.random_state = None

    def __enter__(self):
        # Save current state
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()

        # Set new seed if provided
        if self.seed is not None:
            set_global_seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)
