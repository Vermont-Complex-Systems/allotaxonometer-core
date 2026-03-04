pub mod rtd;
pub mod combine;
pub mod diamond;
pub mod wordshift;
pub mod balance;
pub mod helpers;
pub mod types;

pub use types::*;
pub use combine::comb_elems;
pub use rtd::rank_turbulence_divergence;
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
/// Shares the alpha-independent combElems step across all alphas.
pub fn compute_allotax_multi_alpha(
    system1: &InputSystem,
    system2: &InputSystem,
    alphas: &[f64],
) -> MultiAlphaResult {
    // Step 1: Combine distributions (shared across all alphas)
    let mixed = comb_elems(system1, system2);

    // Step 5: Balance (alpha-independent)
    let bal = balance_dat(system1, system2);

    // Steps 2-4 per alpha
    let alpha_results: Vec<AlphaSlice> = alphas
        .iter()
        .map(|&alpha| {
            let rtd = rank_turbulence_divergence(&mixed, alpha);
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
