use crate::types::{DiamondResult, MixedElements, WordshiftEntry};

/// Generate word shift data for bar chart visualization.
/// Uses the reordered mixed elements from diamond_count (sorted by divergence).
pub fn wordshift_dat(_mixed: &MixedElements, diamond: &DiamondResult) -> Vec<WordshiftEntry> {
    let reordered = &diamond.mixed_elements;
    let deltas = &diamond.deltas;
    let len = reordered.system1.types.len();

    let mut entries: Vec<WordshiftEntry> = Vec::with_capacity(len);

    for i in 0..len {
        let rank_diff = reordered.system1.ranks[i] - reordered.system2.ranks[i];
        let metric = if rank_diff < 0.0 {
            -deltas[i]
        } else {
            deltas[i]
        };

        entries.push(WordshiftEntry {
            type_label: format!(
                "{} ({} \u{21CB} {})",
                reordered.system1.types[i],
                reordered.system1.ranks[i],
                reordered.system2.ranks[i]
            ),
            rank_diff,
            metric,
        });
    }

    // Sort by absolute metric descending
    entries.sort_by(|a, b| {
        b.metric
            .abs()
            .partial_cmp(&a.metric.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    entries
}
