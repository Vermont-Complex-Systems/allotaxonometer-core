use std::collections::HashSet;

use crate::types::{BalanceEntry, InputSystem};

/// Compute balance data between two input systems.
pub fn balance_dat(sys1: &InputSystem, sys2: &InputSystem) -> Vec<BalanceEntry> {
    let types1: HashSet<&str> = sys1.types.iter().map(|s| s.as_str()).collect();
    let types2: HashSet<&str> = sys2.types.iter().map(|s| s.as_str()).collect();

    let union_size = types1.union(&types2).count() as f64;
    let total_types = (types1.len() + types2.len()) as f64;

    let exclusive1 = types1.difference(&types2).count() as f64;
    let exclusive2 = types2.difference(&types1).count() as f64;

    let n1 = types1.len() as f64;
    let n2 = types2.len() as f64;

    vec![
        BalanceEntry {
            y_coord: "total count".to_string(),
            frequency: round3(n2 / total_types),
        },
        BalanceEntry {
            y_coord: "total count".to_string(),
            frequency: round3(-n1 / total_types),
        },
        BalanceEntry {
            y_coord: "all types".to_string(),
            frequency: round3(n2 / union_size),
        },
        BalanceEntry {
            y_coord: "all types".to_string(),
            frequency: round3(-n1 / union_size),
        },
        BalanceEntry {
            y_coord: "exclusive types".to_string(),
            frequency: round3(exclusive2 / n2),
        },
        BalanceEntry {
            y_coord: "exclusive types".to_string(),
            frequency: round3(-exclusive1 / n1),
        },
    ]
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balance_dat() {
        let sys1 = InputSystem {
            types: vec!["a".into(), "b".into(), "c".into()],
            counts: vec![10.0, 5.0, 1.0],
        };
        let sys2 = InputSystem {
            types: vec!["b".into(), "c".into(), "d".into()],
            counts: vec![8.0, 3.0, 2.0],
        };

        let bal = balance_dat(&sys1, &sys2);
        assert_eq!(bal.len(), 6);
        assert_eq!(bal[0].y_coord, "total count");
    }
}
