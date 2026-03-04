use wasm_bindgen::prelude::*;
use allotax_core::{InputSystem, compute_allotax, compute_allotax_multi_alpha};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Backward-compatible: compute RTD only (same signature as the old WASM module).
/// Use this for client-side alpha recomputation on pre-combined data.
#[wasm_bindgen]
pub fn rank_turbulence_divergence(
    ranks1: Vec<f64>,
    ranks2: Vec<f64>,
    counts1: Vec<f64>,
    counts2: Vec<f64>,
    alpha: f64,
) -> JsValue {
    let mixed = allotax_core::MixedElements {
        system1: allotax_core::MixedSystem {
            types: vec![],
            counts: counts1,
            probs: vec![],
            ranks: ranks1,
            totalunique: 0,
        },
        system2: allotax_core::MixedSystem {
            types: vec![],
            counts: counts2,
            probs: vec![],
            ranks: ranks2,
            totalunique: 0,
        },
    };

    let result = allotax_core::rank_turbulence_divergence(&mixed, alpha);

    let json = serde_json::json!({
        "divergence_elements": result.divergence_elements,
        "normalization": result.normalization
    });

    serde_wasm_bindgen::to_value(&json).unwrap()
}

/// Full pipeline: takes two systems (types + counts), returns all visualization data.
#[wasm_bindgen]
pub fn compute_allotax_wasm(
    types1: Vec<String>,
    counts1: Vec<f64>,
    types2: Vec<String>,
    counts2: Vec<f64>,
    alpha: f64,
) -> JsValue {
    let sys1 = InputSystem { types: types1, counts: counts1 };
    let sys2 = InputSystem { types: types2, counts: counts2 };

    let result = compute_allotax(&sys1, &sys2, alpha);

    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Multi-alpha pipeline: precompute for multiple alpha values at once.
#[wasm_bindgen]
pub fn compute_allotax_multi_alpha_wasm(
    types1: Vec<String>,
    counts1: Vec<f64>,
    types2: Vec<String>,
    counts2: Vec<f64>,
    alphas: Vec<f64>,
) -> JsValue {
    let sys1 = InputSystem { types: types1, counts: counts1 };
    let sys2 = InputSystem { types: types2, counts: counts2 };

    let result = compute_allotax_multi_alpha(&sys1, &sys2, &alphas);

    serde_wasm_bindgen::to_value(&result).unwrap()
}
