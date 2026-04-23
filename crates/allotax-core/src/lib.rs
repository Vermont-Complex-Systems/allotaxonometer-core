pub mod rtd;
pub mod combine;
pub mod diamond;
pub mod wordshift;
pub mod balance;
pub mod helpers;
pub mod types;

use rayon::prelude::*;

pub use types::*;
pub use combine::comb_elems;
pub use rtd::{rank_turbulence_divergence, rank_turbulence_divergence_multi_alpha, precompute_inverse_ranks, rank_turbulence_divergence_with_inv};
pub use diamond::diamond_count;
pub use wordshift::wordshift_dat;
pub use balance::balance_dat;

/// Run the full allotaxonometer pipeline:
///   1. Combine distributions (alpha-independent)
///   2. Compute rank-turbulence divergence
///   3. Compute diamond counts
///   4. Compute wordshift data
///   5. Compute balance data
///
/// Returns an AllotaxResult with all structured data ready for visualization.
pub fn compute_allotax(
    system1: &InputSystem,
    system2: &InputSystem,
    alpha: f64,
) -> AllotaxResult {
    // Step 1: Combine distributions (alpha-independent)
    let mixed = comb_elems(system1, system2);

    // Step 2: RTD (alpha-dependent)
    let rtd = rank_turbulence_divergence(&mixed, alpha);

    // Step 3: Diamond counts
    let diamond = diamond_count(&mixed, &rtd);

    // Step 4: Wordshift data
    let wordshift = wordshift_dat(&mixed, &diamond);

    // Step 5: Balance data
    let bal = balance_dat(system1, system2);

    AllotaxResult {
        mixed_elements: mixed,
        rtd,
        diamond,
        wordshift,
        balance: bal,
        alpha,
    }
}

/// Compute allotax for multiple alpha values at once.
/// Shares the alpha-independent combElems step and inverse ranks across all alphas.
/// Per-alpha work is parallelized with Rayon.
pub fn compute_allotax_multi_alpha(
    system1: &InputSystem,
    system2: &InputSystem,
    alphas: &[f64],
) -> MultiAlphaResult {
    // Step 1: Combine distributions (shared across all alphas)
    let mixed = comb_elems(system1, system2);

    // Precompute inverse ranks once (shared across all alphas)
    let (inv_r1, inv_r2) = precompute_inverse_ranks(&mixed);

    // Balance (alpha-independent)
    let bal = balance_dat(system1, system2);

    // Steps 2-4 per alpha — parallelized with Rayon
    let alpha_results: Vec<AlphaSlice> = alphas
        .par_iter()
        .map(|&alpha| {
            let rtd = rank_turbulence_divergence_with_inv(&mixed, &inv_r1, &inv_r2, alpha);
            let diamond = diamond_count(&mixed, &rtd);
            let wordshift = wordshift_dat(&mixed, &diamond);
            AlphaSlice {
                alpha,
                rtd,
                diamond,
                wordshift,
            }
        })
        .collect();

    MultiAlphaResult {
        mixed_elements: mixed,
        balance: bal,
        alpha_results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_systems() -> (InputSystem, InputSystem) {
        let sys1 = InputSystem {
            types: (0..500).map(|i| format!("word_{i}")).collect(),
            counts: (0..500).map(|i| (500 - i) as f64).collect(),
        };
        let sys2 = InputSystem {
            types: (250..750).map(|i| format!("word_{i}")).collect(),
            counts: (0..500).map(|i| (500 - i) as f64).collect(),
        };
        (sys1, sys2)
    }

    #[test]
    fn test_display_wordshift_limit() {
        let (sys1, sys2) = test_systems();
        let result = compute_allotax(&sys1, &sys2, 1.0);

        let full = result.to_display(0);
        let limited = result.to_display(50);

        // Full should have all wordshift entries
        assert!(full.wordshift.len() > 50);
        // Limited should be truncated
        assert_eq!(limited.wordshift.len(), 50);
        // Top entries should be the same
        assert_eq!(full.wordshift[0].type_label, limited.wordshift[0].type_label);
        // Diamond counts should be the same (not affected by limit)
        assert_eq!(full.diamond_counts.len(), limited.diamond_counts.len());
    }

    #[test]
    fn test_multi_alpha_display_wordshift_limit() {
        let (sys1, sys2) = test_systems();
        let alphas = vec![0.5, 1.0, f64::INFINITY];
        let result = compute_allotax_multi_alpha(&sys1, &sys2, &alphas);

        let display = result.to_display(100);
        assert_eq!(display.alpha_results.len(), 3);
        for slice in &display.alpha_results {
            assert!(slice.wordshift.len() <= 100);
        }
    }

    #[test]
    fn test_no_empty_diamond_cells() {
        let (sys1, sys2) = test_systems();
        let result = compute_allotax(&sys1, &sys2, 1.0);

        // All diamond cells should have value > 0
        for cell in &result.diamond.counts {
            assert!(cell.value > 0, "Found empty diamond cell at ({}, {})", cell.x1, cell.y1);
        }
    }

    #[test]
    fn test_multi_alpha_deterministic() {
        let (sys1, sys2) = test_systems();
        let alphas = vec![0.5, 1.0, 2.0];

        let r1 = compute_allotax_multi_alpha(&sys1, &sys2, &alphas);
        let r2 = compute_allotax_multi_alpha(&sys1, &sys2, &alphas);

        // Rayon parallelism should not affect determinism
        for (a, b) in r1.alpha_results.iter().zip(r2.alpha_results.iter()) {
            assert_eq!(a.alpha, b.alpha);
            assert!((a.rtd.delta_sum - b.rtd.delta_sum).abs() < 1e-10);
            assert!((a.rtd.normalization - b.rtd.normalization).abs() < 1e-10);
        }
    }

    #[test]
    fn test_payload_size_with_limit() {
        let (sys1, sys2) = test_systems();
        let alphas = vec![0.5, 1.0, f64::INFINITY];
        let result = compute_allotax_multi_alpha(&sys1, &sys2, &alphas);

        let full_json = serde_json::to_string(&result.to_display(0)).unwrap();
        let limited_json = serde_json::to_string(&result.to_display(50)).unwrap();

        let ratio = limited_json.len() as f64 / full_json.len() as f64;
        println!("Full payload: {} bytes", full_json.len());
        println!("Limited (50) payload: {} bytes", limited_json.len());
        println!("Ratio: {:.2}%", ratio * 100.0);

        assert!(limited_json.len() < full_json.len());
    }
}
