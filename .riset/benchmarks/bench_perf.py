"""Performance benchmark: Encode throughput, query latency p50/p95, and memory usage."""

import os
import shutil
import tempfile
import time
import numpy as np

from lsh_agent_memory.core.index import LSHMemoryIndex


def get_process_memory_mb() -> float:
    """Get current process working set memory in Megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback if psutil is not installed
        return 0.0


def run_performance_benchmark(
    scales: list = [10_000, 50_000],
    dim: int = 384,
    bits: int = 128,
    pool_size: int = 500,
    n_queries: int = 100,
) -> None:
    """Benchmark index build time, RAM usage, and query latency percentiles."""
    print(f"\n=== PERFORMANCE & RESOURCE BENCHMARK (Dim={dim}, Bits={bits}, Pool={pool_size}) ===")

    for n in scales:
        tmp_dir = tempfile.mkdtemp()
        try:
            mem_before = get_process_memory_mb()
            rng = np.random.default_rng(42)

            print(f"\n--- Scale: {n:,} entries ---")
            data = rng.standard_normal((n, dim)).astype(np.float32)
            data /= np.linalg.norm(data, axis=1, keepdims=True)

            index = LSHMemoryIndex(
                dim=dim,
                bits=bits,
                store_dir=tmp_dir,
                pool_size=pool_size,
            )

            # 1. Fit Phase (5k sample)
            sample_fit = data[:min(5_000, n)]
            t_fit0 = time.perf_counter()
            index.fit(sample_fit, iters=50)
            fit_time = (time.perf_counter() - t_fit0) * 1000

            # 2. Encode & Build Phase (target < 2s for 50k)
            t_build0 = time.perf_counter()
            index.add(data)
            build_time = time.perf_counter() - t_build0

            mem_after_build = get_process_memory_mb()
            ram_used = mem_after_build - mem_before

            # 3. Query Latency (p50 / p95)
            query_vecs = rng.standard_normal((n_queries, dim)).astype(np.float32)
            query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

            latencies = []
            for q in query_vecs:
                t0 = time.perf_counter()
                _ = index.query(q, k=10)
                latencies.append((time.perf_counter() - t0) * 1000)  # ms

            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            mean_lat = np.mean(latencies)

            # File sizes on disk
            index_bytes = 0
            if (index.store_dir / "codes.npy").exists():
                index_bytes = (index.store_dir / "codes.npy").stat().st_size
            raw_bytes = (index.store_dir / "vectors.bin").stat().st_size

            print(f"Fit Time (5k sample)      : {fit_time:.2f} ms")
            print(f"Encode & Build Time ({n:,}) : {build_time:.4f} s (Target < 2.0s: {'PASS' if build_time < 2.0 else 'FAIL'})")
            print(f"Index Code Size (RAM/Disk) : {index_bytes / (1024*1024):.2f} MB")
            print(f"Raw Storage Size (Disk)    : {raw_bytes / (1024*1024):.2f} MB")
            print(f"Process Working Set RAM   : ~{ram_used:.2f} MB")
            print(f"Query Latency (p50)        : {p50:.2f} ms")
            print(f"Query Latency (p95)        : {p95:.2f} ms")
            print(f"Query Latency (Mean)       : {mean_lat:.2f} ms")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_performance_benchmark(scales=[10_000, 50_000], dim=384, bits=128)
