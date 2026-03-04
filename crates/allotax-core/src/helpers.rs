/// Compute tied ranks (descending order — highest count = rank 1).
/// Ties get the average rank.
pub fn tiedrank(arr: &[f64]) -> Vec<f64> {
    if arr.is_empty() {
        return vec![];
    }

    // Create (value, original_index) pairs
    let mut indexed: Vec<(f64, usize)> = arr.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();

    // Sort descending by value
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; arr.len()];
    let mut i = 0;
    let n = indexed.len();

    while i < n {
        // Find the extent of the tie group
        let mut j = i + 1;
        while j < n && indexed[j].0 == indexed[i].0 {
            j += 1;
        }

        // Average rank for this tie group (1-based)
        let avg_rank = (i + 1 + j) as f64 / 2.0;

        for k in i..j {
            ranks[indexed[k].1] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Sort array and return sorted values + original indices (like MATLAB sort).
pub fn matlab_sort(arr: &[f64], descending: bool) -> (Vec<f64>, Vec<usize>) {
    if arr.is_empty() {
        return (vec![], vec![]);
    }

    let mut indexed: Vec<(f64, usize)> = arr.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();

    if descending {
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let values = indexed.iter().map(|(v, _)| *v).collect();
    let indices = indexed.iter().map(|(_, i)| *i).collect();

    (values, indices)
}

/// Indices where value > 0.
pub fn which_positive(arr: &[f64]) -> Vec<usize> {
    arr.iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiedrank_simple() {
        let arr = vec![10.0, 5.0, 20.0, 5.0];
        let ranks = tiedrank(&arr);
        // 20 → rank 1, 10 → rank 2, 5 tied → rank 3.5
        assert_eq!(ranks[0], 2.0); // 10
        assert_eq!(ranks[1], 3.5); // 5 (tied)
        assert_eq!(ranks[2], 1.0); // 20
        assert_eq!(ranks[3], 3.5); // 5 (tied)
    }

    #[test]
    fn test_tiedrank_empty() {
        assert_eq!(tiedrank(&[]), Vec::<f64>::new());
    }

    #[test]
    fn test_matlab_sort_descending() {
        let arr = vec![3.0, 1.0, 2.0];
        let (values, indices) = matlab_sort(&arr, true);
        assert_eq!(values, vec![3.0, 2.0, 1.0]);
        assert_eq!(indices, vec![0, 2, 1]);
    }

    #[test]
    fn test_which_positive() {
        let arr = vec![0.0, 5.0, 0.0, 10.0];
        assert_eq!(which_positive(&arr), vec![1, 3]);
    }
}
