"""Micro-benchmark for Hamming bitcode scan optimization."""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def scan_v1_baseline(q_u64: np.ndarray, codes_u64: np.ndarray, n: int):
    # Old implementation
    diff = codes_u64 ^ q_u64
    counts = np.bitwise_count(diff)
    dist = counts.sum(axis=1, dtype=np.uint16)
    part_idx = np.argpartition(dist, n)[:n]
    sub_sort = np.argsort(dist[part_idx])
    return part_idx[sub_sort], dist[part_idx[sub_sort]]

def scan_v2_unrolled(q_u64: np.ndarray, codes_u64: np.ndarray, n: int):
    # Optimized unrolled column implementation
    w = codes_u64.shape[1]
    if w == 2:
        dist = (np.bitwise_count(codes_u64[:, 0] ^ q_u64[0]) +
                np.bitwise_count(codes_u64[:, 1] ^ q_u64[1])).astype(np.uint16)
    elif w == 4:
        dist = (np.bitwise_count(codes_u64[:, 0] ^ q_u64[0]) +
                np.bitwise_count(codes_u64[:, 1] ^ q_u64[1]) +
                np.bitwise_count(codes_u64[:, 2] ^ q_u64[2]) +
                np.bitwise_count(codes_u64[:, 3] ^ q_u64[3])).astype(np.uint16)
    else:
        dist = np.bitwise_count(codes_u64 ^ q_u64).sum(axis=1, dtype=np.uint16)

    part_idx = np.argpartition(dist, n)[:n]
    sub_sort = np.argsort(dist[part_idx])
    return part_idx[sub_sort], dist[part_idx[sub_sort]]

def run_bench():
    print("Benchmarking Hamming bitcode scan optimizations on 1,000,000 128-bit entries...")
    rng = np.random.default_rng(42)
    n_entries = 1_000_000
    codes = rng.integers(0, np.iinfo(np.uint64).max, size=(n_entries, 2), dtype=np.uint64)
    q = rng.integers(0, np.iinfo(np.uint64).max, size=(2,), dtype=np.uint64)
    n_pool = 500

    # Warmup
    scan_v1_baseline(q, codes, n_pool)
    scan_v2_unrolled(q, codes, n_pool)

    # Benchmark V1
    t0 = time.perf_counter()
    n_iter = 20
    for _ in range(n_iter):
        idx1, d1 = scan_v1_baseline(q, codes, n_pool)
    t_v1 = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark V2
    t0 = time.perf_counter()
    for _ in range(n_iter):
        idx2, d2 = scan_v2_unrolled(q, codes, n_pool)
    t_v2 = (time.perf_counter() - t0) / n_iter * 1000

    # Correctness check
    np.testing.assert_array_equal(d1, d2)
    print(f"Results match perfectly!")
    print(f"V1 (2D broadcast + sum(axis=1)): {t_v1:.2f} ms")
    print(f"V2 (1D unrolled column bitwise_count): {t_v2:.2f} ms")
    print(f"Speedup: {t_v1 / t_v2:.2f}x faster ({t_v1 - t_v2:.2f} ms saved per query on 1M entries)!")

if __name__ == "__main__":
    run_bench()
