# GitHub Copilot Instructions

## Project: ltr_wnl (Learning to Rank with Noisy Labels)

Python research package for learning-to-rank under Beta-distributed comparison noise. **Research question**: How does uncertainty in noise rates affect sample complexity?

## Code Conventions

**Type Hints** (Python 3.12+):
- ✅ Use `list[int]`, `dict[str, int]`, `tuple[int, int]` (NOT `List`, `Dict`, `Tuple`)
- ✅ Use `T | None` (NOT `Optional[T]`)
- Rankings: `list[int]` where index 0 = best item

**Imports**:
- ✅ Namespace imports: `import numpy as np`, `from ltr_wnl import noise`
- ✅ Fail loudly: no try/except for missing imports (let `ImportError` raise)

**Data Conventions**:
- Rankings: `[2, 0, 1]` means item 2 is best, item 0 second, item 1 third
- Comparison outcomes: `1` = item i won, `-1` = item j won
- Pairs: always `(i, j)` where `i < j`
- Random state: all functions accept `random_state: int | None` for reproducibility

**Documentation**:
- NumPy-style docstrings with Parameters, Returns, Examples
- Include doctests as executable examples

## Architecture

**Modules**: `noise/` (BetaNoiseModel), `ground_truth/`, `comparisons/` (ComparisonData, ComparisonGenerator), `ranking/` (BordaCount, PlackettLuce, RankCentrality, PageRank), `sampling/` (RandomSampler), `metrics/` (Kendall's tau, top-k), `simulation/` (SimulationRunner)

**Design**: Modular with abstract base classes (`NoiseModel`, `Ranker`, `SamplerStrategy`)

## Extending the Package

**Ranker**: Inherit from `Ranker`, implement `fit(comparison_data: ComparisonData) -> list[int]`

```python
from ltr_wnl.ranking.ranker_base import Ranker
from ltr_wnl.comparisons.data import ComparisonData

class MyRanker(Ranker):
    def fit(self, comparison_data: ComparisonData) -> list[int]:
        # Return list of item indices from best to worst
        return ranking
```

**Noise Model**: Inherit from `NoiseModel`, implement `sample_error_rate(i, j) -> float`

```python
from ltr_wnl.noise.noise_base import NoiseModel

class MyNoiseModel(NoiseModel):
    def sample_error_rate(self, i: int | None = None, j: int | None = None) -> float:
        return error_rate  # in [0, 1]
```

## Key Concepts

- **BTL Model**: `P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j))`
- **Beta Noise**: Error probability `p_ij ~ Beta(μφ, (1-μ)φ)` where `μ` = mean error, `φ` = concentration (low φ = high uncertainty)

## Common Patterns

```python
from ltr_wnl import noise, ranking, sampling, simulation

runner = simulation.SimulationRunner(
    n_items=100,
    noise_model=noise.BetaNoiseModel(mu=0.3, phi=2.0),
    ranker=ranking.PlackettLuceRanker(),
    sampling_strategy=sampling.RandomSampler()
)

results = runner.run_trial(n_comparisons=500, random_state=42)
```

## Anti-Patterns

❌ Don't reimplement choix methods - use the library
❌ Don't break reproducibility - always use seeded RNGs
❌ Don't mix up ranking conventions - rankings are item indices, not scores
❌ Don't forget pair ordering - always `(i, j)` where `i < j`

---

See [CLAUDE.md](../CLAUDE.md) for full development guide
