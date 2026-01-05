"""
Active sampling strategies for selecting comparison pairs.
"""

from ltr_wnl.sampling.strategy_base import SamplingStrategy
from ltr_wnl.sampling.random import RandomSampling

__all__ = ["SamplingStrategy", "RandomSampling"]
