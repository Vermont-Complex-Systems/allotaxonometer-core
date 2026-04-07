# allotax

Python bindings for [allotax-core](https://crates.io/crates/allotax-core) — fast allotaxonometer computation powered by Rust via PyO3.

## Install

```bash
pip install allotax
```

## Usage

```python
import allotax

sys1 = {"types": ["word_a", "word_b", "word_c"], "counts": [100.0, 50.0, 25.0]}
sys2 = {"types": ["word_b", "word_c", "word_d"], "counts": [80.0, 40.0, 20.0]}

# Single alpha — lean display result
result = allotax.compute_allotax(sys1, sys2, alpha=1.0)
# result keys: normalization, delta_sum, diamond_counts, max_delta_loss, wordshift, balance, alpha

# Multiple alphas at once (shared combElems step, Rayon-parallelized)
result = allotax.compute_allotax_multi_alpha(sys1, sys2, alphas=[0.5, 1.0, float("inf")])
# result keys: balance, alpha_results

# Full intermediate data (for custom downstream processing)
result = allotax.compute_allotax_full(sys1, sys2, alpha=1.0)
result = allotax.compute_allotax_multi_alpha_full(sys1, sys2, alphas=[0.5, 1.0])
```

## API

| Function | Returns |
|---|---|
| `compute_allotax(sys1, sys2, alpha, wordshift_limit=200)` | Lean display dict for a single α |
| `compute_allotax_full(sys1, sys2, alpha)` | Full intermediate result for a single α |
| `compute_allotax_multi_alpha(sys1, sys2, alphas, wordshift_limit=200)` | Lean display dict for multiple α values |
| `compute_allotax_multi_alpha_full(sys1, sys2, alphas)` | Full intermediate result for multiple α values |
| `rank_turbulence_divergence(ranks1, ranks2, counts1, counts2, alpha)` | RTD only (pre-combined data) |

Input systems are plain dicts with `"types"` (list of str) and `"counts"` (list of float).

## References

Gallagher et al. (2021). [Generalized word shift graphs](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-021-00260-3). *EPJ Data Science*, 10(1), 4.
