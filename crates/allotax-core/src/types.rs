use serde::{Deserialize, Serialize};

/// Input system: just types and counts. Probs/totalunique are calculated if missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSystem {
    pub types: Vec<String>,
    pub counts: Vec<f64>,
}

/// A single system after combining distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedSystem {
    pub types: Vec<String>,
    pub counts: Vec<f64>,
    pub probs: Vec<f64>,
    pub ranks: Vec<f64>,
    pub totalunique: usize,
}

/// Pair of mixed systems (result of combElems).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedElements {
    pub system1: MixedSystem,
    pub system2: MixedSystem,
}

/// Result of rank-turbulence divergence computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtdResult {
    pub divergence_elements: Vec<f64>,
    pub normalization: f64,
    /// Sum of normalized divergence elements = actual D_alpha^R value.
    pub delta_sum: f64,
}

/// A single cell in the diamond plot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiamondCell {
    pub x1: i32,
    pub y1: i32,
    pub coord_on_diag: f64,
    pub cos_dist: f64,
    pub rank: String,
    #[serde(rename = "rank_L")]
    pub rank_l: Vec<f64>,
    #[serde(rename = "rank_R")]
    pub rank_r: Vec<f64>,
    pub value: usize,
    pub types: String,
    pub which_sys: String,
}

/// Result of diamond_count computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiamondResult {
    pub counts: Vec<DiamondCell>,
    pub deltas: Vec<f64>,
    pub max_delta_loss: f64,
    /// Reordered mixed elements (sorted by divergence contribution).
    pub mixed_elements: MixedElements,
    /// Number of cells along one side of the diamond grid. Computed from
    /// `maxlog10` during `diamond_count` — stored here so downstream
    /// consumers don't have to re-derive it from the ranks.
    pub ncells: i32,
    /// Largest log10(rank) across both systems, rounded up to at least 1.
    pub maxlog10: f64,
}

/// A single entry in the wordshift bar chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordshiftEntry {
    #[serde(rename = "type")]
    pub type_label: String,
    pub rank_diff: f64,
    pub metric: f64,
}

/// A single entry in the balance data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceEntry {
    pub y_coord: String,
    pub frequency: f64,
}

/// Full result for a single alpha computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllotaxResult {
    pub mixed_elements: MixedElements,
    pub rtd: RtdResult,
    pub diamond: DiamondResult,
    pub wordshift: Vec<WordshiftEntry>,
    pub balance: Vec<BalanceEntry>,
    pub alpha: f64,
}

/// Per-alpha slice (for multi-alpha precomputation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaSlice {
    pub alpha: f64,
    pub rtd: RtdResult,
    pub diamond: DiamondResult,
    pub wordshift: Vec<WordshiftEntry>,
}

/// Result when computing for multiple alpha values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAlphaResult {
    pub mixed_elements: MixedElements,
    pub balance: Vec<BalanceEntry>,
    pub alpha_results: Vec<AlphaSlice>,
}

// --- Display-only types (lean output for API/Python) ---

/// Lean diamond cell for API responses.
/// Drops: `rank` (unused), `rank_R` (unused), replaces `rank_L` with single `max_rank`.
/// Empty cells (value=0) are filtered out entirely in `to_display()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayDiamondCell {
    pub x1: i32,
    pub y1: i32,
    pub value: usize,
    pub types: String,
    pub which_sys: String,
    pub coord_on_diag: f64,
    pub cos_dist: f64,
    /// max of rank_L (was rank_L[1]) — only value the frontend reads.
    pub max_rank: f64,
}

/// Lean wordshift entry — drops `rank_diff` (never rendered, sign already in `metric`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayWordshiftEntry {
    #[serde(rename = "type")]
    pub type_label: String,
    pub metric: f64,
}

/// Lean result for API responses — only what the frontend needs to render.
/// Empty diamond cells filtered out, unused fields stripped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllotaxDisplayResult {
    pub normalization: f64,
    /// Sum of normalized divergence elements = actual D_alpha^R value.
    pub delta_sum: f64,
    pub diamond_counts: Vec<DisplayDiamondCell>,
    pub max_delta_loss: f64,
    pub wordshift: Vec<DisplayWordshiftEntry>,
    pub balance: Vec<BalanceEntry>,
    pub alpha: f64,
    /// Number of cells along one side of the diamond grid. The frontend uses
    /// this to size its band scale when consuming sparse `diamond_counts`.
    pub ncells: i32,
    /// Largest log10(rank) across both systems, rounded up to at least 1.
    /// Used by the frontend to label the diamond axes.
    pub maxlog10: f64,
}

/// Lean per-alpha slice for multi-alpha API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaDisplaySlice {
    pub alpha: f64,
    pub normalization: f64,
    /// Sum of normalized divergence elements = actual D_alpha^R value.
    pub delta_sum: f64,
    pub diamond_counts: Vec<DisplayDiamondCell>,
    pub max_delta_loss: f64,
    pub wordshift: Vec<DisplayWordshiftEntry>,
    /// Number of cells along one side of the diamond grid. Same value across
    /// all alphas (it depends only on `mixed_elements`), but copied per slice
    /// for convenience.
    pub ncells: i32,
    /// Largest log10(rank) across both systems, rounded up to at least 1.
    pub maxlog10: f64,
}

/// Lean multi-alpha result for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAlphaDisplayResult {
    pub balance: Vec<BalanceEntry>,
    pub alpha_results: Vec<AlphaDisplaySlice>,
}

/// Truncate the comma-separated types string to at most `max_types` entries.
/// The frontend only shows the first 8 in tooltips — no need to send thousands.
fn truncate_types(types: &str, max_types: usize) -> String {
    let mut count = 0;
    for (i, c) in types.char_indices() {
        if c == ',' {
            count += 1;
            if count >= max_types {
                return types[..i].to_string();
            }
        }
    }
    types.to_string()
}

fn to_display_diamond(cells: &[DiamondCell]) -> Vec<DisplayDiamondCell> {
    cells.iter()
        .filter(|c| c.value > 0)
        .map(|c| DisplayDiamondCell {
            x1: c.x1,
            y1: c.y1,
            value: c.value,
            types: truncate_types(&c.types, 10),
            which_sys: c.which_sys.clone(),
            coord_on_diag: c.coord_on_diag,
            cos_dist: c.cos_dist,
            max_rank: c.rank_l.get(1).copied().unwrap_or(0.0),
        })
        .collect()
}

fn to_display_wordshift(entries: &[WordshiftEntry], limit: usize) -> Vec<DisplayWordshiftEntry> {
    let iter = entries.iter();
    let iter: Box<dyn Iterator<Item = &WordshiftEntry>> = if limit > 0 {
        Box::new(iter.take(limit))
    } else {
        Box::new(iter)
    };
    iter.map(|e| DisplayWordshiftEntry {
            type_label: e.type_label.clone(),
            metric: e.metric,
        })
        .collect()
}

impl AllotaxResult {
    /// Convert to a lean display result, dropping intermediate data.
    /// `wordshift_limit`: max wordshift entries per alpha (0 = no limit).
    pub fn to_display(&self, wordshift_limit: usize) -> AllotaxDisplayResult {
        AllotaxDisplayResult {
            normalization: self.rtd.normalization,
            delta_sum: self.rtd.delta_sum,
            diamond_counts: to_display_diamond(&self.diamond.counts),
            max_delta_loss: self.diamond.max_delta_loss,
            wordshift: to_display_wordshift(&self.wordshift, wordshift_limit),
            balance: self.balance.clone(),
            alpha: self.alpha,
            ncells: self.diamond.ncells,
            maxlog10: self.diamond.maxlog10,
        }
    }
}

impl MultiAlphaResult {
    /// Convert to a lean display result, dropping intermediate data.
    /// `wordshift_limit`: max wordshift entries per alpha (0 = no limit).
    pub fn to_display(&self, wordshift_limit: usize) -> MultiAlphaDisplayResult {
        MultiAlphaDisplayResult {
            balance: self.balance.clone(),
            alpha_results: self.alpha_results.iter().map(|a| AlphaDisplaySlice {
                alpha: a.alpha,
                normalization: a.rtd.normalization,
                delta_sum: a.rtd.delta_sum,
                diamond_counts: to_display_diamond(&a.diamond.counts),
                max_delta_loss: a.diamond.max_delta_loss,
                wordshift: to_display_wordshift(&a.wordshift, wordshift_limit),
                ncells: a.diamond.ncells,
                maxlog10: a.diamond.maxlog10,
            }).collect(),
        }
    }
}
