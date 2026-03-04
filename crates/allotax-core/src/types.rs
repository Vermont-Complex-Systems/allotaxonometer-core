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
}

/// A single cell in the diamond plot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiamondCell {
    pub x1: i32,
    pub y1: i32,
    pub coord_on_diag: f64,
    pub cos_dist: f64,
    pub rank: String,
    pub rank_l: Vec<f64>,
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
