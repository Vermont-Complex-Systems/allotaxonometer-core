use pyo3::prelude::*;

use ::allotax_core as core;
use core::InputSystem;

/// Convert an InputSystem from Python dicts with "types" and "counts" keys.
fn parse_system(obj: &Bound<'_, PyAny>) -> PyResult<InputSystem> {
    let types: Vec<String> = obj.get_item("types")?.extract()?;
    let counts: Vec<f64> = obj.get_item("counts")?.extract()?;
    Ok(InputSystem { types, counts })
}

/// Serialize any serde-serializable struct to a Python dict.
fn to_py_dict<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<PyObject> {
    let json_str = serde_json::to_string(value)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let json_mod = py.import("json")?;
    let result = json_mod.call_method1("loads", (json_str,))?;
    Ok(result.into())
}

/// Compute allotaxonometer for a single alpha value.
/// Returns lean display data (no intermediate computation arrays).
///
/// Args:
///     system1: dict with "types" (list[str]) and "counts" (list[float])
///     system2: dict with "types" (list[str]) and "counts" (list[float])
///     alpha: float (divergence parameter)
///     wordshift_limit: int, optional (max wordshift entries; default 200, 0 = no limit)
///
/// Returns:
///     dict with keys: normalization, diamond_counts, max_delta_loss, wordshift, balance, alpha
#[pyfunction]
#[pyo3(signature = (system1, system2, alpha, wordshift_limit=200))]
fn compute_allotax(
    py: Python<'_>,
    system1: &Bound<'_, PyAny>,
    system2: &Bound<'_, PyAny>,
    alpha: f64,
    wordshift_limit: usize,
) -> PyResult<PyObject> {
    let sys1 = parse_system(system1)?;
    let sys2 = parse_system(system2)?;
    let result = core::compute_allotax(&sys1, &sys2, alpha);
    to_py_dict(py, &result.to_display(wordshift_limit))
}

/// Compute allotaxonometer for a single alpha, returning the full result
/// including intermediate data (mixed_elements, divergence_elements, deltas, etc.).
///
/// Args:
///     system1: dict with "types" (list[str]) and "counts" (list[float])
///     system2: dict with "types" (list[str]) and "counts" (list[float])
///     alpha: float (divergence parameter)
///
/// Returns:
///     dict with keys: mixed_elements, rtd, diamond, wordshift, balance, alpha
#[pyfunction]
fn compute_allotax_full(
    py: Python<'_>,
    system1: &Bound<'_, PyAny>,
    system2: &Bound<'_, PyAny>,
    alpha: f64,
) -> PyResult<PyObject> {
    let sys1 = parse_system(system1)?;
    let sys2 = parse_system(system2)?;
    let result = core::compute_allotax(&sys1, &sys2, alpha);
    to_py_dict(py, &result)
}

/// Compute allotaxonometer for multiple alpha values.
/// Returns lean display data. Shares the alpha-independent step (comb_elems) across all alphas.
///
/// Args:
///     system1: dict with "types" and "counts"
///     system2: dict with "types" and "counts"
///     alphas: list[float]
///     wordshift_limit: int, optional (max wordshift entries per alpha; default 200, 0 = no limit)
///
/// Returns:
///     dict with keys: balance, alpha_results
#[pyfunction]
#[pyo3(signature = (system1, system2, alphas, wordshift_limit=200))]
fn compute_allotax_multi_alpha(
    py: Python<'_>,
    system1: &Bound<'_, PyAny>,
    system2: &Bound<'_, PyAny>,
    alphas: Vec<f64>,
    wordshift_limit: usize,
) -> PyResult<PyObject> {
    let sys1 = parse_system(system1)?;
    let sys2 = parse_system(system2)?;
    let result = core::compute_allotax_multi_alpha(&sys1, &sys2, &alphas);
    to_py_dict(py, &result.to_display(wordshift_limit))
}

/// Compute allotaxonometer for multiple alphas, returning the full result.
///
/// Args:
///     system1: dict with "types" and "counts"
///     system2: dict with "types" and "counts"
///     alphas: list[float]
///
/// Returns:
///     dict with keys: mixed_elements, balance, alpha_results
#[pyfunction]
fn compute_allotax_multi_alpha_full(
    py: Python<'_>,
    system1: &Bound<'_, PyAny>,
    system2: &Bound<'_, PyAny>,
    alphas: Vec<f64>,
) -> PyResult<PyObject> {
    let sys1 = parse_system(system1)?;
    let sys2 = parse_system(system2)?;
    let result = core::compute_allotax_multi_alpha(&sys1, &sys2, &alphas);
    to_py_dict(py, &result)
}

/// Per-term RTD entry with structured fields.
#[derive(serde::Serialize)]
struct RtdEntry {
    #[serde(rename = "type")]
    type_label: String,
    rank1: f64,
    rank2: f64,
    divergence: f64,
}

/// Full RTD result: per-term entries + summary stats.
#[derive(serde::Serialize)]
struct RtdWordshiftResult {
    wordshift: Vec<RtdEntry>,
    normalization: f64,
    delta_sum: f64,
}

/// Per-alpha RTD result with wordshift entries.
#[derive(serde::Serialize)]
struct AlphaRtdSlice {
    alpha: f64,
    wordshift: Vec<RtdEntry>,
    normalization: f64,
    delta_sum: f64,
}

/// Multi-alpha RTD result.
#[derive(serde::Serialize)]
struct MultiAlphaRtdResult {
    alpha_results: Vec<AlphaRtdSlice>,
}

/// Compute rank-turbulence divergence between two systems.
///
/// Takes two systems (dicts with "types" and "counts"), properly merges
/// their vocabularies via comb_elems, computes RTD, and returns per-term
/// signed divergence with ranks.
///
/// Args:
///     system1: dict with "types" (list[str]) and "counts" (list[float])
///     system2: dict with "types" (list[str]) and "counts" (list[float])
///     alpha: float (divergence parameter)
///     limit: int, optional (max entries to return; default 0 = no limit)
///
/// Returns:
///     dict with keys:
///       - wordshift: list of {type, rank1, rank2, divergence}
///       - normalization: float
///       - delta_sum: float
#[pyfunction]
#[pyo3(signature = (system1, system2, alpha, limit=0))]
fn rank_turbulence_divergence(
    py: Python<'_>,
    system1: &Bound<'_, PyAny>,
    system2: &Bound<'_, PyAny>,
    alpha: f64,
    limit: usize,
) -> PyResult<PyObject> {
    let sys1 = parse_system(system1)?;
    let sys2 = parse_system(system2)?;
    let mixed = core::comb_elems(&sys1, &sys2);
    let rtd = core::rank_turbulence_divergence(&mixed, alpha);

    let n = mixed.system1.types.len();
    let mut entries: Vec<RtdEntry> = Vec::with_capacity(n);

    for i in 0..n {
        let rank1 = mixed.system1.ranks[i];
        let rank2 = mixed.system2.ranks[i];
        let rank_diff = rank1 - rank2;
        // Sign: negative rank_diff means rank1 < rank2 (type is more
        // prominent in system1), so flip the unsigned element.
        let divergence = if rank_diff < 0.0 {
            -rtd.divergence_elements[i]
        } else {
            rtd.divergence_elements[i]
        };
        entries.push(RtdEntry {
            type_label: mixed.system1.types[i].clone(),
            rank1,
            rank2,
            divergence,
        });
    }

    // Sort by |divergence| descending
    entries.sort_by(|a, b| {
        b.divergence.abs().partial_cmp(&a.divergence.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if limit > 0 {
        entries.truncate(limit);
    }

    let result = RtdWordshiftResult {
        wordshift: entries,
        normalization: rtd.normalization,
        delta_sum: rtd.delta_sum,
    };
    to_py_dict(py, &result)
}

/// Compute rank-turbulence divergence for multiple alpha values.
/// Shares vocabulary merging and inverse rank precomputation across all alphas.
/// Per-alpha work is parallelized with Rayon.
///
/// Args:
///     system1: dict with "types" (list[str]) and "counts" (list[float])
///     system2: dict with "types" (list[str]) and "counts" (list[float])
///     alphas: list[float] (divergence parameters)
///     limit: int, optional (max entries per alpha; default 0 = no limit)
///
/// Returns:
///     dict with keys:
///       - alpha_results: list of {alpha, wordshift, normalization, delta_sum}
#[pyfunction]
#[pyo3(signature = (system1, system2, alphas, limit=0))]
fn rank_turbulence_divergence_multi_alpha(
    py: Python<'_>,
    system1: &Bound<'_, PyAny>,
    system2: &Bound<'_, PyAny>,
    alphas: Vec<f64>,
    limit: usize,
) -> PyResult<PyObject> {
    let sys1 = parse_system(system1)?;
    let sys2 = parse_system(system2)?;
    let mixed = core::comb_elems(&sys1, &sys2);
    let rtd_results = core::rank_turbulence_divergence_multi_alpha(&mixed, &alphas);

    let n = mixed.system1.types.len();
    let mut alpha_slices: Vec<AlphaRtdSlice> = Vec::with_capacity(rtd_results.len());

    for (alpha, rtd) in rtd_results {
        let mut entries: Vec<RtdEntry> = Vec::with_capacity(n);
        for i in 0..n {
            let rank1 = mixed.system1.ranks[i];
            let rank2 = mixed.system2.ranks[i];
            let rank_diff = rank1 - rank2;
            let divergence = if rank_diff < 0.0 {
                -rtd.divergence_elements[i]
            } else {
                rtd.divergence_elements[i]
            };
            entries.push(RtdEntry {
                type_label: mixed.system1.types[i].clone(),
                rank1,
                rank2,
                divergence,
            });
        }
        entries.sort_by(|a, b| {
            b.divergence.abs().partial_cmp(&a.divergence.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if limit > 0 {
            entries.truncate(limit);
        }
        alpha_slices.push(AlphaRtdSlice {
            alpha,
            wordshift: entries,
            normalization: rtd.normalization,
            delta_sum: rtd.delta_sum,
        });
    }

    let result = MultiAlphaRtdResult {
        alpha_results: alpha_slices,
    };
    to_py_dict(py, &result)
}

/// Python module: `import allotax`
#[pymodule]
fn allotax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_allotax, m)?)?;
    m.add_function(wrap_pyfunction!(compute_allotax_full, m)?)?;
    m.add_function(wrap_pyfunction!(compute_allotax_multi_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(compute_allotax_multi_alpha_full, m)?)?;
    m.add_function(wrap_pyfunction!(rank_turbulence_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rank_turbulence_divergence_multi_alpha, m)?)?;
    Ok(())
}
