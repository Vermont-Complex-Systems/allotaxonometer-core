use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use allotax_core::{
    comb_elems, compute_allotax, compute_allotax_multi_alpha, diamond_count,
    rank_turbulence_divergence, InputSystem,
};

/// Generate a Zipf-like dataset with `n` types.
/// Mimics real word frequency distributions (like the JS performance tests).
fn generate_zipf_system(n: usize, seed: u64) -> InputSystem {
    let mut types = Vec::with_capacity(n);
    let mut counts = Vec::with_capacity(n);

    for i in 0..n {
        // Simple deterministic "random" offset so two systems differ
        let rank = (i + 1) as f64;
        let count = (1000.0 / rank.powf(0.8)) + ((i as f64 * seed as f64) % 7.0);
        types.push(format!("type_{}", i + seed as usize * 100_000));
        counts.push(count.max(1.0));
    }

    InputSystem { types, counts }
}

/// Generate two overlapping systems (shared ~70% of types, like real data).
fn generate_pair(n: usize) -> (InputSystem, InputSystem) {
    let shared = (n as f64 * 0.7) as usize;
    let exclusive = n - shared;

    let mut types1 = Vec::with_capacity(n);
    let mut counts1 = Vec::with_capacity(n);
    let mut types2 = Vec::with_capacity(n);
    let mut counts2 = Vec::with_capacity(n);

    // Shared types
    for i in 0..shared {
        let rank = (i + 1) as f64;
        let word = format!("word_{i}");
        types1.push(word.clone());
        counts1.push((1000.0 / rank.powf(0.8)).max(1.0));
        types2.push(word);
        counts2.push((800.0 / rank.powf(0.75)).max(1.0));
    }

    // Exclusive to system 1
    for i in 0..exclusive {
        types1.push(format!("excl1_{i}"));
        counts1.push(((exclusive - i) as f64).max(1.0));
    }

    // Exclusive to system 2
    for i in 0..exclusive {
        types2.push(format!("excl2_{i}"));
        counts2.push(((exclusive - i) as f64).max(1.0));
    }

    (
        InputSystem { types: types1, counts: counts1 },
        InputSystem { types: types2, counts: counts2 },
    )
}

fn bench_comb_elems(c: &mut Criterion) {
    let mut group = c.benchmark_group("comb_elems");

    for size in [1_000, 10_000, 100_000] {
        let (sys1, sys2) = generate_pair(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| comb_elems(black_box(&sys1), black_box(&sys2)))
        });
    }

    group.finish();
}

fn bench_rtd(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_turbulence_divergence");

    for size in [1_000, 10_000, 100_000] {
        let (sys1, sys2) = generate_pair(size);
        let mixed = comb_elems(&sys1, &sys2);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| rank_turbulence_divergence(black_box(&mixed), black_box(1.0)))
        });
    }

    group.finish();
}

fn bench_diamond_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("diamond_count");

    for size in [1_000, 10_000] {
        let (sys1, sys2) = generate_pair(size);
        let mixed = comb_elems(&sys1, &sys2);
        let rtd = rank_turbulence_divergence(&mixed, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| diamond_count(black_box(&mixed), black_box(&rtd)))
        });
    }

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    for size in [1_000, 10_000, 100_000] {
        let (sys1, sys2) = generate_pair(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| compute_allotax(black_box(&sys1), black_box(&sys2), black_box(1.0)))
        });
    }

    group.finish();
}

fn bench_multi_alpha(c: &mut Criterion) {
    let (sys1, sys2) = generate_pair(10_000);
    let alphas = vec![0.5, 1.0, 2.0, 3.0, f64::INFINITY];

    c.bench_function("multi_alpha_10k_5alphas", |b| {
        b.iter(|| {
            compute_allotax_multi_alpha(black_box(&sys1), black_box(&sys2), black_box(&alphas))
        })
    });
}

criterion_group!(
    benches,
    bench_comb_elems,
    bench_rtd,
    bench_diamond_count,
    bench_full_pipeline,
    bench_multi_alpha,
);
criterion_main!(benches);
