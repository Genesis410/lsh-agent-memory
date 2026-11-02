"""04_benchmark_demo.py - Quick CPU benchmark for LSH vs Flat Cosine."""

import time
import numpy as np
from lsh_agent_memory import LSHMemoryIndex

def main():
    print("=== LSH-Agent-Memory CPU Performance Demo ===")
    n_vectors = 50_000
    dim = 384
    bits = 128

    print(f"Generating {n_vectors:,} random 384-dimensional synthetic vectors...")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    print("\n1. Exact NumPy BLAS Flat Cosine Search:")
    query = X[0].copy()
    t0 = time.perf_counter()
    exact_scores = X @ query
    top10_exact = np.argsort(-exact_scores)[:10]
    t_exact = (time.perf_counter() - t0) * 1000
    print(f"   - Latency: {t_exact:.2f} ms")
    print(f"   - Memory footprint (Float32 matrix): {X.nbytes / (1024*1024):.2f} MB")

    print("\n2. Initializing LSHMemoryIndex (128-bit ITQ):")
    index = LSHMemoryIndex(
        dim=dim,
        bits=bits,
        store_dir="./bench_demo_store",
        pool_size=500,
    )
    t0 = time.perf_counter()
    index.fit(X[:2500])  # Fast calibration on 2,500 samples
    index.add(X)
    t_build = time.perf_counter() - t0
    print(f"   - Index build & encode time: {t_build:.2f} s")
    print(f"   - Bitcode RAM footprint (16 bytes/vec): {index.codes.nbytes / (1024*1024):.2f} MB")
    print(f"   - RAM savings vs Flat: {(1.0 - index.codes.nbytes / X.nbytes) * 100:.1f}%")

    print("\n3. LSH Hamming Scan + Cosine Re-rank Search:")
    t0 = time.perf_counter()
    results = index.query(query, k=10)
    t_lsh = (time.perf_counter() - t0) * 1000
    lsh_indices = [r["index"] for r in results]
    overlap = len(set(top10_exact).intersection(lsh_indices)) / 10.0 * 100.0

    print(f"   - Latency: {t_lsh:.2f} ms")
    print(f"   - Top-10 Cosine Overlap: {overlap:.1f}%")
    print("\n[SUCCESS] Benchmark demo complete.")

if __name__ == "__main__":
    main()
