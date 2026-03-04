use std::collections::HashMap;

use crate::helpers::matlab_sort;
use crate::types::{DiamondCell, DiamondResult, MixedElements, MixedSystem, RtdResult};

const CELL_LENGTH: f64 = 1.0 / 15.0;

fn rank2coord(rank: f64) -> i32 {
    (rank.log10() / CELL_LENGTH).floor() as i32
}

fn rank_maxlog10(mixed: &MixedElements) -> f64 {
    let max1 = mixed
        .system1
        .ranks
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let max2 = mixed
        .system2
        .ranks
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    max1.log10().max(max2.log10()).ceil()
}

/// Compute diamond counts from mixed elements.
/// Also reorders mixed elements by divergence contribution (descending).
pub fn diamond_count(mixed: &MixedElements, rtd: &RtdResult) -> DiamondResult {
    let deltas = &rtd.divergence_elements;

    // Sort by divergence (descending) to get reordering indices
    let (_sorted_vals, indices) = matlab_sort(deltas, true);

    let len = indices.len();

    // Reorder everything by divergence contribution
    let mut reordered_deltas = Vec::with_capacity(len);
    let mut types = Vec::with_capacity(len);
    let mut counts1 = Vec::with_capacity(len);
    let mut counts2 = Vec::with_capacity(len);
    let mut ranks1 = Vec::with_capacity(len);
    let mut ranks2 = Vec::with_capacity(len);
    let mut probs1 = Vec::with_capacity(len);
    let mut probs2 = Vec::with_capacity(len);

    for &idx in &indices {
        reordered_deltas.push(deltas[idx]);
        types.push(mixed.system1.types[idx].clone());
        counts1.push(mixed.system1.counts[idx]);
        counts2.push(mixed.system2.counts[idx]);
        ranks1.push(mixed.system1.ranks[idx]);
        ranks2.push(mixed.system2.ranks[idx]);
        probs1.push(mixed.system1.probs[idx]);
        probs2.push(mixed.system2.probs[idx]);
    }

    // Compute deltas_loss for max_delta_loss
    let mut max_delta_loss = f64::NEG_INFINITY;
    for i in 0..len {
        let d = if ranks1[i] > ranks2[i] {
            -1.0
        } else {
            reordered_deltas[i]
        };
        if d > max_delta_loss {
            max_delta_loss = d;
        }
    }

    // Build reordered mixed elements
    let reordered_mixed = MixedElements {
        system1: MixedSystem {
            types: types.clone(),
            counts: counts1,
            probs: probs1,
            ranks: ranks1,
            totalunique: len,
        },
        system2: MixedSystem {
            types,
            counts: counts2,
            probs: probs2,
            ranks: ranks2,
            totalunique: len,
        },
    };

    // Build diamond grid
    let mut maxlog10 = rank_maxlog10(&reordered_mixed);
    if maxlog10 < 1.0 {
        maxlog10 = 1.0;
    }
    let ncells = (maxlog10 / CELL_LENGTH).floor() as i32 + 1;

    // Group items by coordinate
    let mut coord_groups: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for i in 0..len {
        let x = rank2coord(reordered_mixed.system2.ranks[i]);
        let y = rank2coord(reordered_mixed.system1.ranks[i]);
        coord_groups.entry((x, y)).or_default().push(i);
    }

    // Build cell grid
    let mut counts = Vec::with_capacity((ncells * ncells) as usize);
    for i in 0..ncells {
        for j in 0..ncells {
            let key = (i, j);
            let which_sys = if i - j <= 0 { "right" } else { "left" };

            match coord_groups.get(&key) {
                None => {
                    counts.push(DiamondCell {
                        x1: i,
                        y1: j,
                        coord_on_diag: (j + i) as f64 / 2.0,
                        cos_dist: ((i - j) * (i - j)) as f64,
                        rank: String::new(),
                        rank_l: vec![],
                        rank_r: vec![],
                        value: 0,
                        types: String::new(),
                        which_sys: which_sys.to_string(),
                    });
                }
                Some(items) => {
                    let first = items[0];
                    let rank_str = format!(
                        "({}, {})",
                        reordered_mixed.system1.ranks[first],
                        reordered_mixed.system2.ranks[first]
                    );

                    let rank_l_vals: Vec<f64> =
                        items.iter().map(|&i| reordered_mixed.system1.ranks[i]).collect();
                    let rank_r_vals: Vec<f64> =
                        items.iter().map(|&i| reordered_mixed.system2.ranks[i]).collect();

                    let min_l = rank_l_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_l = rank_l_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let min_r = rank_r_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_r = rank_r_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let types_str: String = items
                        .iter()
                        .map(|&i| reordered_mixed.system1.types[i].as_str())
                        .collect::<Vec<_>>()
                        .join(", ");

                    counts.push(DiamondCell {
                        x1: i,
                        y1: j,
                        coord_on_diag: (j + i) as f64 / 2.0,
                        cos_dist: ((i - j) * (i - j)) as f64,
                        rank: rank_str,
                        rank_l: vec![min_l, max_l],
                        rank_r: vec![min_r, max_r],
                        value: items.len(),
                        types: types_str,
                        which_sys: which_sys.to_string(),
                    });
                }
            }
        }
    }

    DiamondResult {
        counts,
        deltas: reordered_deltas,
        max_delta_loss,
        mixed_elements: reordered_mixed,
    }
}
