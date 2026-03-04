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
///
/// Args:
///     system1: dict with "types" (list[str]) and "counts" (list[float])
///     system2: dict with "types" (list[str]) and "counts" (list[float])
///     alpha: float (divergence parameter)
///
/// Returns:
///     dict with keys: mixed_elements, rtd, diamond, wordshift, balance, alpha
#[pyfunction]
fn compute_allotax(
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
/// Shares the alpha-independent step (comb_elems) across all alphas.
///
/// Args:
///     system1: dict with "types" and "counts"
///     system2: dict with "types" and "counts"
///     alphas: list[float]
///
/// Returns:
///     dict with keys: mixed_elements, balance, alpha_results
#[pyfunction]
fn compute_allotax_multi_alpha(
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

/// Compute only the rank-turbulence divergence (for recomputing on pre-combined data).
#[pyfunction]
fn rank_turbulence_divergence(
    py: Python<'_>,
    ranks1: Vec<f64>,
    ranks2: Vec<f64>,
    counts1: Vec<f64>,
    counts2: Vec<f64>,
    alpha: f64,
) -> PyResult<PyObject> {
    let mixed = core::MixedElements {
        system1: core::MixedSystem {
            types: vec![],
            counts: counts1,
            probs: vec![],
            ranks: ranks1,
            totalunique: 0,
        },
        system2: core::MixedSystem {
            types: vec![],
            counts: counts2,
            probs: vec![],
            ranks: ranks2,
            totalunique: 0,
        },
    };

    let result = core::rank_turbulence_divergence(&mixed, alpha);
    to_py_dict(py, &result)
}

/// Python module: `import allotax`
#[pymodule]
fn allotax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_allotax, m)?)?;
    m.add_function(wrap_pyfunction!(compute_allotax_multi_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(rank_turbulence_divergence, m)?)?;
    Ok(())
}
