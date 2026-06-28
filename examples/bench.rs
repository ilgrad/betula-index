//! Build / query / size comparison of `betula-index` against the `std` maps it competes with.
//!
//! Run with `cargo run --release --example bench` (release matters — `lto` + `opt-level=3`). Numbers
//! are illustrative and machine-dependent; the *ratios* are the point. No external dependencies: a
//! deterministic stride walks the keys in a non-sequential order so lookups are not pure cache hits,
//! and a checksum is printed so the optimiser cannot elide the queries.

use betula_index::{PerfectHashIndex, StringIndex};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

fn bench<T>(label: &str, n: usize, build: impl FnOnce() -> T, query: impl Fn(&T) -> u64) {
    let t0 = Instant::now();
    let s = build();
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Two passes: warm, then timed; report ns per lookup.
    let _ = query(&s);
    let t1 = Instant::now();
    let checksum = query(&s);
    let per = t1.elapsed().as_secs_f64() * 1e9 / n as f64;
    println!("{label:24} build {build_ms:8.1} ms   lookup {per:6.1} ns/op   (checksum {checksum})");
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    // Zero-padded so lexicographic order is the natural order (fair to the ordered structures).
    let keys: Vec<String> = (0..n).map(|i| format!("entity-{i:012}")).collect();
    // Non-sequential, full-coverage probe order (stride by a large coprime of n... use a prime step).
    const STEP: usize = 0x9E37_79B1; // ~golden-ratio odd constant → coprime with any power-of-two-ish n
    let probe: Vec<usize> = (0..n).map(|i| (i.wrapping_mul(STEP)) % n).collect();

    println!(
        "betula-index bench — n = {n} keys (len {} each)\n",
        keys[0].len()
    );

    bench(
        "betula StringIndex (FST)",
        n,
        || StringIndex::build(&keys).unwrap(),
        |idx| probe.iter().map(|&i| idx.id(&keys[i]).unwrap_or(0)).sum(),
    );
    bench(
        "betula PerfectHashIndex",
        n,
        || PerfectHashIndex::build(&keys).unwrap(),
        |idx| {
            probe
                .iter()
                .map(|&i| idx.id(&keys[i]).map_or(0, u64::from))
                .sum()
        },
    );
    bench(
        "std HashMap<String,u32>",
        n,
        || {
            keys.iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), i as u32))
                .collect::<HashMap<_, _>>()
        },
        |m| {
            probe
                .iter()
                .map(|&i| m.get(&keys[i]).copied().map_or(0, u64::from))
                .sum()
        },
    );
    bench(
        "std BTreeMap<String,u32>",
        n,
        || {
            keys.iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), i as u32))
                .collect::<BTreeMap<_, _>>()
        },
        |m| {
            probe
                .iter()
                .map(|&i| m.get(&keys[i]).copied().map_or(0, u64::from))
                .sum()
        },
    );

    // Serialised size: the build-once / query-many payload you persist or mmap.
    let si = StringIndex::build(&keys).unwrap();
    let ph = PerfectHashIndex::build(&keys).unwrap();
    let raw = keys.iter().map(|k| k.len()).sum::<usize>();
    println!("\nserialised size (bytes/key):");
    println!(
        "  betula StringIndex blob   {:6.2}",
        si.to_bytes().len() as f64 / n as f64
    );
    println!(
        "  betula PerfectHashIndex   {:6.2}",
        ph.to_bytes().unwrap().len() as f64 / n as f64
    );
    println!("  raw key bytes (no index)  {:6.2}", raw as f64 / n as f64);

    // A capability the maps do not have: typo-tolerant + prefix queries over the same structure.
    let t = Instant::now();
    let fuzzy = si.fuzzy("entity-000000000042", 1).unwrap().len();
    let pfx = si.prefix("entity-00000000000").len();
    println!(
        "\nStringIndex extras: fuzzy(d=1) hit {fuzzy} match(es), prefix hit {pfx}, in {:.2} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
}
