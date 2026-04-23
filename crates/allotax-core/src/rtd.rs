use crate::helpers::which_positive;
use crate::types::{MixedElements, RtdResult};

/// Compute divergence elements for given inverse ranks and alpha.
fn div_elems(inv_r1: &[f64], inv_r2: &[f64], alpha: f64) -> Vec<f64> {
    if alpha == f64::INFINITY {
        inv_r1
            .iter()
            .zip(inv_r2.iter())
            .map(|(r1, r2)| {
                if r1 == r2 {
                    0.0
                } else {
                    r1.max(*r2)
                }
            })
            .collect()
    } else if alpha == 0.0 {
        inv_r1
            .iter()
            .zip(inv_r2.iter())
            .map(|(r1, r2)| {
                let x_max = (1.0 / r1).max(1.0 / r2);
                let x_min = (1.0 / r1).min(1.0 / r2);
                (x_max / x_min).ln()
            })
            .collect()
    } else {
        inv_r1
            .iter()
            .zip(inv_r2.iter())
            .map(|(r1, r2)| {
                ((alpha + 1.0) / alpha)
                    * (r1.powf(alpha) - r2.powf(alpha))
                        .abs()
                        .powf(1.0 / (alpha + 1.0))
            })
            .collect()
    }
}

/// Compute the normalization factor for divergence elements.
fn norm_div_elems(
    counts1: &[f64],
    counts2: &[f64],
    inv_r1: &[f64],
    inv_r2: &[f64],
    alpha: f64,
) -> f64 {
    let indices1 = which_positive(counts1);
    let indices2 = which_positive(counts2);

    let n1 = indices1.len() as f64;
    let n2 = indices2.len() as f64;

    let inv_r1_disjoint = 1.0 / (n2 + n1 / 2.0);
    let inv_r2_disjoint = 1.0 / (n1 + n2 / 2.0);

    if alpha == f64::INFINITY {
        let sum1: f64 = indices1.iter().map(|&i| inv_r1[i]).sum();
        let sum2: f64 = indices2.iter().map(|&i| inv_r2[i]).sum();
        sum1 + sum2
    } else if alpha == 0.0 {
        let term1: f64 = indices1
            .iter()
            .map(|&i| (inv_r1[i] / inv_r2_disjoint).ln().abs())
            .sum();
        let term2: f64 = indices2
            .iter()
            .map(|&i| (inv_r2[i] / inv_r1_disjoint).ln().abs())
            .sum();
        term1 + term2
    } else {
        let term1: f64 = ((alpha + 1.0) / alpha)
            * indices1
                .iter()
                .map(|&i| {
                    (inv_r1[i].powf(alpha).abs() - inv_r2_disjoint.powf(alpha))
                        .powf(1.0 / (alpha + 1.0))
                })
                .sum::<f64>();

        let term2: f64 = ((alpha + 1.0) / alpha)
            * indices2
                .iter()
                .map(|&i| {
                    (inv_r1_disjoint.powf(alpha) - inv_r2[i].powf(alpha))
                        .abs()
                        .powf(1.0 / (alpha + 1.0))
                })
                .sum::<f64>();

        term1 + term2
    }
}

/// Precompute inverse ranks from mixed elements (call once, reuse across alphas).
pub fn precompute_inverse_ranks(mixed: &MixedElements) -> (Vec<f64>, Vec<f64>) {
    let inv_r1: Vec<f64> = mixed.system1.ranks.iter().map(|r| r.powi(-1)).collect();
    let inv_r2: Vec<f64> = mixed.system2.ranks.iter().map(|r| r.powi(-1)).collect();
    (inv_r1, inv_r2)
}

/// RTD with pre-computed inverse ranks (avoids recomputing per alpha).
pub fn rank_turbulence_divergence_with_inv(
    mixed: &MixedElements,
    inv_r1: &[f64],
    inv_r2: &[f64],
    alpha: f64,
) -> RtdResult {
    let divergence_elements = div_elems(inv_r1, inv_r2, alpha);
    let normalization = norm_div_elems(
        &mixed.system1.counts,
        &mixed.system2.counts,
        inv_r1,
        inv_r2,
        alpha,
    );

    let normalized: Vec<f64> = divergence_elements
        .iter()
        .map(|d| d / normalization)
        .collect();

    let delta_sum: f64 = normalized.iter().sum();

    RtdResult {
        divergence_elements: normalized,
        normalization,
        delta_sum,
    }
}

/// Compute rank-turbulence divergence between two mixed systems.
pub fn rank_turbulence_divergence(mixed: &MixedElements, alpha: f64) -> RtdResult {
    let (inv_r1, inv_r2) = precompute_inverse_ranks(mixed);
    rank_turbulence_divergence_with_inv(mixed, &inv_r1, &inv_r2, alpha)
}

/// Compute rank-turbulence divergence for multiple alpha values.
/// Precomputes inverse ranks once, then parallelizes per-alpha work with Rayon.
pub fn rank_turbulence_divergence_multi_alpha(
    mixed: &MixedElements,
    alphas: &[f64],
) -> Vec<(f64, RtdResult)> {
    use rayon::prelude::*;
    let (inv_r1, inv_r2) = precompute_inverse_ranks(mixed);
    alphas
        .par_iter()
        .map(|&alpha| {
            let rtd = rank_turbulence_divergence_with_inv(mixed, &inv_r1, &inv_r2, alpha);
            (alpha, rtd)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_elems_infinity() {
        let inv_r1 = vec![1.0, 0.5, 0.33];
        let inv_r2 = vec![1.0, 0.33, 0.5];
        let result = div_elems(&inv_r1, &inv_r2, f64::INFINITY);

        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.5);
        assert_eq!(result[2], 0.5);
    }

    #[test]
    fn test_div_elems_alpha_1() {
        let inv_r1 = vec![1.0, 0.5];
        let inv_r2 = vec![0.5, 1.0];
        let result = div_elems(&inv_r1, &inv_r2, 1.0);

        // (2/1) * |1^1 - 0.5^1|^(1/2) = 2 * 0.5^0.5 = 2 * 0.7071 ≈ 1.4142
        assert!((result[0] - 2.0 * 0.5_f64.sqrt()).abs() < 1e-10);
    }
}
