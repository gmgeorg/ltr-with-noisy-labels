# Claude Development Guide

`ltr_wnl` is a Python research package for learning-to-rank algorithms under Beta-distributed comparison noise. **Research question**: How does uncertainty in noise rates affect sample complexity?

## Planning & Specs

All planning documents, design specs, and work-in-progress notes should go in the **`/specs/`** directory (gitignored). Use this for:

- Experiment designs and research plans
- Feature specifications
- Implementation notes and TODO lists
- Performance analysis and benchmarking results

## Key Concepts

- **BTL Model**: `P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j))`
- **Beta Noise**: Error probability `p_ij ~ Beta(μφ, (1-μ)φ)` where `μ` = mean error, `φ` = concentration (low φ = high uncertainty)
- **Hypothesis**: Low concentration requires more samples for same ranking accuracy

## Architecture

**Design**: Modular with abstract base classes (`NoiseModel`, `Ranker`, `SamplerStrategy`), reproducible via `random_state`, integrates with `choix`

**Modules**: `noise/` (BetaNoiseModel), `ground_truth/`, `comparisons/` (ComparisonData, ComparisonGenerator), `ranking/` (BordaCount, PlackettLuce, RankCentrality), `sampling/` (RandomSampler), `metrics/` (Kendall's tau, top-k), `simulation/` (SimulationRunner)

**Reference**: See [notebooks/pl-model.ipynb](notebooks/pl-model.ipynb) for original implementation

## Extending the Package

**Noise Model**: Inherit from `NoiseModel`, implement `sample_error_rate(i, j) -> float`
**Ranker**: Inherit from `Ranker`, implement `fit(comparison_data: ComparisonData) -> list[int]`
**Sampler**: Inherit from `SamplerStrategy`, implement selection logic

## Code Conventions

**Type Hints** (Python 3.12+):

- Use `list[int]`, `dict[str, int]`, `tuple[int, int]` (NOT `List`, `Dict`, `Tuple` from `typing`)
- Use `T | None` (NOT `Optional[T]`)
- Rankings: `list[int]` where index 0 = best item, values = item indices

**Imports**:

- Use namespace imports: `import numpy as np`, `from ltr_wnl import noise`
- Fail loudly: no try/except for missing imports (let `ImportError` raise)

**Data Conventions**:

- Rankings: `[2, 0, 1]` means item 2 is best, item 0 second, item 1 third
- Comparison outcomes: `1` = item i won, `-1` = item j won
- Pairs: always `(i, j)` where `i < j`
- Random state: all functions accept `random_state: int | None` for reproducibility

**Documentation**:

- NumPy-style docstrings with Parameters, Returns, Examples
- Include doctests as executable examples

## Common Issues

- **Module not found**: Run `pip install -e .`
- **choix import error**: Install choix separately
- **Numerical issues**: Small `phi` (< 1e-6) causes Beta sampling issues; large strength differences need log-sum-exp trick
- **Comparison retrieval**: Use `ComparisonData.get_outcomes(i, j)` to handle pair ordering automatically
