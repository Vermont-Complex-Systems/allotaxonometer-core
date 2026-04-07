# allotax-core

Core computation engine for the [allotaxonometer](https://github.com/Vermont-Complex-Systems/allotaxonometer-core) — a tool for comparing two ranked lists using rank-turbulence divergence (RTD).

## What it does

Given two frequency distributions (types + counts), it computes:

- **Rank-turbulence divergence** (D_α^R) — a tunable divergence parameterized by α
- **Diamond counts** — 2D histogram of types by their ranks in each system
- **Wordshift data** — per-type contribution to the divergence, sorted by magnitude
- **Balance data** — aggregate frequency comparison between the two systems

The `α` parameter controls sensitivity to rank differences at the top vs. tail of the distribution. `α = 1` weights all ranks equally; `α → ∞` focuses on the top ranks only.

## Usage

```rust
use allotax_core::{InputSystem, compute_allotax, compute_allotax_multi_alpha};

let sys1 = InputSystem {
    types: vec!["word_a".into(), "word_b".into()],
    counts: vec![100.0, 50.0],
};
let sys2 = InputSystem {
    types: vec!["word_b".into(), "word_c".into()],
    counts: vec![80.0, 30.0],
};

// Single alpha
let result = compute_allotax(&sys1, &sys2, 1.0);
let display = result.to_display(200); // top-200 wordshift entries

// Multiple alphas (shared combElems step, Rayon-parallelized)
let result = compute_allotax_multi_alpha(&sys1, &sys2, &[0.5, 1.0, f64::INFINITY]);
let display = result.to_display(200);
```

## Python bindings

See [`allotax`](https://pypi.org/project/allotax/) for the PyO3-based Python package.

## References

Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J., Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for visualizing and explaining pairwise comparisons between texts](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-021-00260-3). *EPJ Data Science*, 10(1), 4.
