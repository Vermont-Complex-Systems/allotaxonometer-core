use std::collections::HashMap;

use crate::helpers::tiedrank;
use crate::types::{InputSystem, MixedElements, MixedSystem};

/// Combine two input systems into a unified representation.
/// This is the alpha-independent step — computes the union of types,
/// aligned counts/probs, and tied ranks.
pub fn comb_elems(sys1: &InputSystem, sys2: &InputSystem) -> MixedElements {
    // Normalize: compute probs from counts
    let total1: f64 = sys1.counts.iter().sum();
    let total2: f64 = sys2.counts.iter().sum();

    // Build union via HashMap (single pass, like the optimized JS version)
    let mut type_map: HashMap<&str, (f64, f64, f64, f64)> = HashMap::new();

    for (i, t) in sys1.types.iter().enumerate() {
        let prob = if total1 > 0.0 { sys1.counts[i] / total1 } else { 0.0 };
        type_map.insert(t.as_str(), (sys1.counts[i], prob, 0.0, 0.0));
    }

    for (i, t) in sys2.types.iter().enumerate() {
        let prob = if total2 > 0.0 { sys2.counts[i] / total2 } else { 0.0 };
        type_map
            .entry(t.as_str())
            .and_modify(|e| {
                e.2 = sys2.counts[i];
                e.3 = prob;
            })
            .or_insert((0.0, 0.0, sys2.counts[i], prob));
    }

    let len = type_map.len();
    let mut types = Vec::with_capacity(len);
    let mut counts1 = Vec::with_capacity(len);
    let mut counts2 = Vec::with_capacity(len);
    let mut probs1 = Vec::with_capacity(len);
    let mut probs2 = Vec::with_capacity(len);

    for (t, (c1, p1, c2, p2)) in &type_map {
        types.push(t.to_string());
        counts1.push(*c1);
        probs1.push(*p1);
        counts2.push(*c2);
        probs2.push(*p2);
    }

    let ranks1 = tiedrank(&counts1);
    let ranks2 = tiedrank(&counts2);

    MixedElements {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comb_elems_basic() {
        let sys1 = InputSystem {
            types: vec!["a".into(), "b".into(), "c".into()],
            counts: vec![10.0, 5.0, 1.0],
        };
        let sys2 = InputSystem {
            types: vec!["b".into(), "c".into(), "d".into()],
            counts: vec![8.0, 3.0, 2.0],
        };

        let mixed = comb_elems(&sys1, &sys2);

        // Union should have 4 types: a, b, c, d
        assert_eq!(mixed.system1.types.len(), 4);
        assert_eq!(mixed.system2.types.len(), 4);
        assert_eq!(mixed.system1.totalunique, 4);

        // Check that 'a' has 0 count in system2
        let a_idx = mixed.system1.types.iter().position(|t| t == "a").unwrap();
        assert_eq!(mixed.system2.counts[a_idx], 0.0);

        // Check that 'd' has 0 count in system1
        let d_idx = mixed.system1.types.iter().position(|t| t == "d").unwrap();
        assert_eq!(mixed.system1.counts[d_idx], 0.0);
    }
}
