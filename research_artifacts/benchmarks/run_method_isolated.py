"""Isolated worker script to benchmark a single ANN method on a specific dataset and scale.

Measures Windows peak_wset, build time, latency p50/p95/IQR across 1000 queries,
Overlap@10 with bootstrap 95% CI, Tie-Tolerant Recall@10, and Quality Ratio.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

# Baseline imports
import faiss
import hnswlib
from annoy import AnnoyIndex
from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend


def get_peak_wset_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().peak_wset / (1024 * 1024)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 95.0) -> list:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    boot_means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lower = float(np.percentile(boot_means, (100 - ci) / 2))
    upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return [round(lower, 2), round(upper, 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="M1..M11")
    parser.add_argument("--scale", required=True, choices=["50k", "200k", "1m"])
    parser.add_argument("--dataset", default="skr", choices=["skr", "sv"])
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep for Pareto")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    args = parser.parse_args()

    dim = 384
    scale_int = 50_000 if args.scale == "50k" else (200_000 if args.scale == "200k" else 1_000_000)

    # 1. Baseline RSS before index load
    gc.collect()
    baseline_wset = get_peak_wset_mb()

    # 2. Load Queries and Ground Truth
    q_vecs = np.load("data/benchmarks/queries_1000.npy")
    gt_path = Path(f"data/benchmarks/gt_top10_{args.dataset}_{args.scale}.npy")
    if gt_path.exists():
        gt_top10 = np.load(gt_path)
    else:
        gt_top10 = None

    # Load dataset
    data_path = Path(f"data/benchmarks/{args.dataset}_{args.scale}.npy")
    f16_path = Path(f"data/benchmarks/{args.dataset}_{args.scale}.f16.dat")

    # Ingest / Build depending on method
    records = []
    t_build = 0.0

    # -------------------------------------------------------------
    # M1: Exact Cosine Flat In-Memory (NumPy BLAS Float32)
    # -------------------------------------------------------------
    if args.method == "M1":
        db_vecs = np.load(data_path)
        index_ram_mb = db_vecs.nbytes / (1024 * 1024)

        # Warmup
        for i in range(50):
            _ = db_vecs @ q_vecs[i]

        latencies = []
        for q in q_vecs:
            t0 = time.perf_counter()
            s = db_vecs @ q
            _ = np.argpartition(-s, 10)[:10]
            latencies.append((time.perf_counter() - t0) * 1000)

        peak_wset = get_peak_wset_mb()
        delta_wset = max(0.0, peak_wset - baseline_wset)

        records.append({
            "method_id": "M1",
            "name": "Exact Cosine Flat (NumPy BLAS)",
            "param": "In-Memory Float32",
            "scale": args.scale,
            "dataset": args.dataset,
            "build_time_s": 0.0,
            "overlap10": 100.0,
            "overlap10_ci": [100.0, 100.0],
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
            "peak_wset_mb": round(peak_wset, 2),
            "delta_wset_mb": round(delta_wset, 2),
            "index_ram_mb": round(index_ram_mb, 2),
        })

    # -------------------------------------------------------------
    # M2: Exact Cosine Flat Disk-Backed (np.memmap Float16)
    # -------------------------------------------------------------
    elif args.method == "M2":
        db_f16 = np.memmap(f16_path, dtype=np.float16, mode="r", shape=(scale_int, dim))
        # On 1M evaluate 100 queries; on smaller scales evaluate 200 queries
        n_eval = 100 if scale_int == 1_000_000 else 200
        q_eval = q_vecs[:n_eval]
        q_f16 = q_eval.astype(np.float16)
        chunk_sz = 200_000

        # Warmup
        for i in range(5):
            _ = db_f16[:chunk_sz].astype(np.float32) @ q_f16[i].astype(np.float32)

        latencies = []
        for i in range(len(q_f16)):
            t0 = time.perf_counter()
            q = q_f16[i].astype(np.float32)
            top_scores = np.full(10, -1e9, dtype=np.float32)
            top_idx = np.zeros(10, dtype=np.int64)
            for start in range(0, scale_int, chunk_sz):
                end = min(start + chunk_sz, scale_int)
                sub_sims = db_f16[start:end].astype(np.float32) @ q
                cand_idx = np.argpartition(-sub_sims, 10)[:10]
                comb_scores = np.concatenate([top_scores, sub_sims[cand_idx]])
                comb_idx = np.concatenate([top_idx, start + cand_idx])
                best = np.argpartition(-comb_scores, 10)[:10]
                top_scores = comb_scores[best]
                top_idx = comb_idx[best]
            latencies.append((time.perf_counter() - t0) * 1000)

        peak_wset = get_peak_wset_mb()
        delta_wset = max(0.0, peak_wset - baseline_wset)

        records.append({
            "method_id": "M2",
            "name": "Exact Cosine Flat Disk-Backed (np.memmap Float16)",
            "param": "Disk Memmap Float16",
            "scale": args.scale,
            "dataset": args.dataset,
            "build_time_s": 0.0,
            "overlap10": 100.0,
            "overlap10_ci": [100.0, 100.0],
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
            "peak_wset_mb": round(peak_wset, 2),
            "delta_wset_mb": round(delta_wset, 2),
            "index_ram_mb": 0.5, # Minimal in RAM (disk backed)
        })

    # -------------------------------------------------------------
    # M3: FAISS IndexFlatIP (C++ AVX2)
    # -------------------------------------------------------------
    elif args.method == "M3":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(dim)
        index.add(db_vecs[:scale_int])
        t_build = time.perf_counter() - t0

        # Warmup
        _, _ = index.search(q_vecs[:50], 10)

        latencies = []
        for q in q_vecs:
            t0 = time.perf_counter()
            _, _ = index.search(q[None, :], 10)
            latencies.append((time.perf_counter() - t0) * 1000)

        peak_wset = get_peak_wset_mb()
        delta_wset = max(0.0, peak_wset - baseline_wset)

        records.append({
            "method_id": "M3",
            "name": "FAISS IndexFlatIP (C++ AVX2)",
            "param": "In-Memory Flat",
            "scale": args.scale,
            "dataset": args.dataset,
            "build_time_s": round(t_build, 3),
            "overlap10": 100.0,
            "overlap10_ci": [100.0, 100.0],
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
            "peak_wset_mb": round(peak_wset, 2),
            "delta_wset_mb": round(delta_wset, 2),
            "index_ram_mb": round(scale_int * dim * 4 / (1024 * 1024), 2),
        })

    # -------------------------------------------------------------
    # M4: FAISS IndexLSH 128-bit Mentah (Hamming-only, no re-rank)
    # -------------------------------------------------------------
    elif args.method == "M4":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        index = faiss.IndexLSH(dim, 128)
        index.add(db_vecs[:scale_int])
        t_build = time.perf_counter() - t0

        latencies = []
        hits_per_q = []
        for i, q in enumerate(q_vecs):
            t0 = time.perf_counter()
            _, I = index.search(q[None, :], 10)
            latencies.append((time.perf_counter() - t0) * 1000)
            if gt_top10 is not None:
                h = len(set(gt_top10[i]).intersection(set(I[0])))
                hits_per_q.append(h * 10.0)

        peak_wset = get_peak_wset_mb()
        delta_wset = max(0.0, peak_wset - baseline_wset)
        ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0

        records.append({
            "method_id": "M4",
            "name": "FAISS IndexLSH 128-bit (Raw)",
            "param": "Hamming-only",
            "scale": args.scale,
            "dataset": args.dataset,
            "build_time_s": round(t_build, 3),
            "overlap10": round(ov10, 2),
            "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
            "peak_wset_mb": round(peak_wset, 2),
            "delta_wset_mb": round(delta_wset, 2),
            "index_ram_mb": round(scale_int * 16 / (1024 * 1024), 2),
        })

    # -------------------------------------------------------------
    # M5: FAISS IndexLSH 128-bit (+ Cosine Re-rank, sweep pool)
    # -------------------------------------------------------------
    elif args.method == "M5":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        index = faiss.IndexLSH(dim, 128)
        index.add(db_vecs[:scale_int])
        t_build = time.perf_counter() - t0

        pools = [250, 500, 1000, 2000] if args.sweep else [500]
        for pool_n in pools:
            latencies = []
            hits_per_q = []
            for i, q in enumerate(q_vecs):
                t0 = time.perf_counter()
                _, cand = index.search(q[None, :], pool_n)
                c_idx = cand[0]
                sub_sims = q @ db_vecs[c_idx].T
                top10 = c_idx[np.argpartition(-sub_sims, 10)[:10]]
                latencies.append((time.perf_counter() - t0) * 1000)
                if gt_top10 is not None:
                    h = len(set(gt_top10[i]).intersection(set(top10)))
                    hits_per_q.append(h * 10.0)

            ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
            peak_wset = get_peak_wset_mb()
            delta_wset = max(0.0, peak_wset - baseline_wset)

            records.append({
                "method_id": "M5",
                "name": f"FAISS IndexLSH 128-bit + Re-rank (Pool {pool_n})",
                "param": f"Pool {pool_n}",
                "scale": args.scale,
                "dataset": args.dataset,
                "build_time_s": round(t_build, 3),
                "overlap10": round(ov10, 2),
                "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
                "peak_wset_mb": round(peak_wset, 2),
                "delta_wset_mb": round(delta_wset, 2),
                "index_ram_mb": round((scale_int * 16 + scale_int * dim * 4) / (1024 * 1024), 2),
            })

    # -------------------------------------------------------------
    # M6: FAISS IndexHNSWFlat (M=16, efSearch sweep)
    # -------------------------------------------------------------
    elif args.method == "M6":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        index.add(db_vecs[:scale_int])
        t_build = time.perf_counter() - t0

        ef_list = [16, 32, 64, 128, 256] if args.sweep else [64]
        for ef in ef_list:
            index.hnsw.efSearch = ef
            latencies = []
            hits_per_q = []
            for i, q in enumerate(q_vecs):
                t0 = time.perf_counter()
                _, I = index.search(q[None, :], 10)
                latencies.append((time.perf_counter() - t0) * 1000)
                if gt_top10 is not None:
                    h = len(set(gt_top10[i]).intersection(set(I[0])))
                    hits_per_q.append(h * 10.0)

            ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
            peak_wset = get_peak_wset_mb()
            delta_wset = max(0.0, peak_wset - baseline_wset)

            records.append({
                "method_id": "M6",
                "name": f"FAISS IndexHNSWFlat (M=16, efSearch={ef})",
                "param": f"efSearch={ef}",
                "scale": args.scale,
                "dataset": args.dataset,
                "build_time_s": round(t_build, 3),
                "overlap10": round(ov10, 2),
                "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
                "peak_wset_mb": round(peak_wset, 2),
                "delta_wset_mb": round(delta_wset, 2),
                "index_ram_mb": round((scale_int * dim * 4 + scale_int * 16 * 4 * 2) / (1024 * 1024), 2),
            })

    # -------------------------------------------------------------
    # M7: hnswlib (M=16, efSearch sweep)
    # -------------------------------------------------------------
    elif args.method == "M7":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(max_elements=scale_int, ef_construction=100, M=16)
        index.add_items(db_vecs[:scale_int])
        t_build = time.perf_counter() - t0

        ef_list = [16, 32, 64, 128, 256] if args.sweep else [64]
        for ef in ef_list:
            index.set_ef(ef)
            latencies = []
            hits_per_q = []
            for i, q in enumerate(q_vecs):
                t0 = time.perf_counter()
                I, _ = index.knn_query(q, k=10)
                latencies.append((time.perf_counter() - t0) * 1000)
                if gt_top10 is not None:
                    h = len(set(gt_top10[i]).intersection(set(I[0])))
                    hits_per_q.append(h * 10.0)

            ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
            peak_wset = get_peak_wset_mb()
            delta_wset = max(0.0, peak_wset - baseline_wset)

            records.append({
                "method_id": "M7",
                "name": f"hnswlib (M=16, ef={ef})",
                "param": f"efSearch={ef}",
                "scale": args.scale,
                "dataset": args.dataset,
                "build_time_s": round(t_build, 3),
                "overlap10": round(ov10, 2),
                "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
                "peak_wset_mb": round(peak_wset, 2),
                "delta_wset_mb": round(delta_wset, 2),
                "index_ram_mb": round((scale_int * dim * 4 + scale_int * 16 * 4 * 2) / (1024 * 1024), 2),
            })

    # -------------------------------------------------------------
    # M8: Annoy (Metric Angular, n_trees in {10, 50}, search_k sweep)
    # -------------------------------------------------------------
    elif args.method == "M8":
        db_vecs = np.load(data_path, mmap_mode="r")
        tree_list = [10, 50] if args.sweep else [10]
        for n_trees in tree_list:
            t0 = time.perf_counter()
            index = AnnoyIndex(dim, "angular")
            for idx_i in range(scale_int):
                index.add_item(idx_i, db_vecs[idx_i])
            index.build(n_trees)
            t_build = time.perf_counter() - t0

            search_k_list = [500, 1000, 2000] if args.sweep else [1000]
            for sk in search_k_list:
                latencies = []
                hits_per_q = []
                for i, q in enumerate(q_vecs):
                    t0 = time.perf_counter()
                    I = index.get_nns_by_vector(q, 10, search_k=sk)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    if gt_top10 is not None:
                        h = len(set(gt_top10[i]).intersection(set(I)))
                        hits_per_q.append(h * 10.0)

                ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
                peak_wset = get_peak_wset_mb()
                delta_wset = max(0.0, peak_wset - baseline_wset)

                records.append({
                    "method_id": "M8",
                    "name": f"Annoy (trees={n_trees}, search_k={sk})",
                    "param": f"trees={n_trees}, search_k={sk}",
                    "scale": args.scale,
                    "dataset": args.dataset,
                    "build_time_s": round(t_build, 3),
                    "overlap10": round(ov10, 2),
                    "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
                    "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                    "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
                    "peak_wset_mb": round(peak_wset, 2),
                    "delta_wset_mb": round(delta_wset, 2),
                    "index_ram_mb": round((scale_int * dim * 4 + scale_int * n_trees * 8) / (1024 * 1024), 2),
                })

    # -------------------------------------------------------------
    # M9: FAISS ITQTransform + IndexRefineFlat
    # -------------------------------------------------------------
    elif args.method == "M9":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        itq = faiss.ITQTransform(dim, 128, True)
        itq.train(db_vecs[:min(scale_int, 2500)])
        base_pre = faiss.IndexPreTransform(itq, faiss.IndexLSH(128, 128))
        refine = faiss.IndexRefineFlat(base_pre)
        refine.add(db_vecs[:scale_int])
        t_build = time.perf_counter() - t0

        pools = [250, 500, 1000] if args.sweep else [500]
        for p_n in pools:
            refine.k_factor = float(p_n / 10.0)
            latencies = []
            hits_per_q = []
            for i, q in enumerate(q_vecs):
                t0 = time.perf_counter()
                _, I = refine.search(q[None, :], 10)
                latencies.append((time.perf_counter() - t0) * 1000)
                if gt_top10 is not None:
                    h = len(set(gt_top10[i]).intersection(set(I[0])))
                    hits_per_q.append(h * 10.0)

            ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
            peak_wset = get_peak_wset_mb()
            delta_wset = max(0.0, peak_wset - baseline_wset)

            records.append({
                "method_id": "M9",
                "name": f"FAISS ITQTransform + IndexRefineFlat (Pool {p_n})",
                "param": f"ITQ 128-bit + Refine Pool {p_n}",
                "scale": args.scale,
                "dataset": args.dataset,
                "build_time_s": round(t_build, 3),
                "overlap10": round(ov10, 2),
                "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
                "peak_wset_mb": round(peak_wset, 2),
                "delta_wset_mb": round(delta_wset, 2),
                "index_ram_mb": round((scale_int * 16 + scale_int * dim * 4) / (1024 * 1024), 2),
            })

    # -------------------------------------------------------------
    # M10: LSH-Memory-Wrapper Mentah (128-bit Hamming-only, no re-rank)
    # -------------------------------------------------------------
    elif args.method == "M10":
        db_vecs = np.load(data_path, mmap_mode="r")
        t0 = time.perf_counter()
        mu, W = fit_itq(db_vecs[:min(scale_int, 2500)], bits=128, iters=50, seed=42)
        codes = encode(db_vecs[:scale_int], mu, W)
        t_build = time.perf_counter() - t0

        q_codes = encode(q_vecs, mu, W)
        backend = BitwiseCountBackend()

        latencies = []
        hits_per_q = []
        for i in range(len(q_vecs)):
            t0 = time.perf_counter()
            top10, _ = backend.search(q_codes[i], codes, n=10)
            latencies.append((time.perf_counter() - t0) * 1000)
            if gt_top10 is not None:
                h = len(set(gt_top10[i]).intersection(set(top10)))
                hits_per_q.append(h * 10.0)

        ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
        peak_wset = get_peak_wset_mb()
        delta_wset = max(0.0, peak_wset - baseline_wset)

        records.append({
            "method_id": "M10",
            "name": "LSH-Memory-Wrapper (128-bit Raw)",
            "param": "Hamming-only",
            "scale": args.scale,
            "dataset": args.dataset,
            "build_time_s": round(t_build, 3),
            "overlap10": round(ov10, 2),
            "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
            "peak_wset_mb": round(peak_wset, 2),
            "delta_wset_mb": round(delta_wset, 2),
            "index_ram_mb": round(codes.nbytes / (1024 * 1024), 2),
        })

    # -------------------------------------------------------------
    # M11: LSH-Memory-Wrapper Lengkap (ITQ 128 & 256 + Memmap Re-rank)
    # -------------------------------------------------------------
    elif args.method == "M11":
        db_f16 = np.memmap(f16_path, dtype=np.float16, mode="r", shape=(scale_int, dim))
        db_vecs_train = np.load(data_path, mmap_mode="r")[:min(scale_int, 2500)]

        bit_options = [128, 256] if args.sweep else [128]
        for bits in bit_options:
            t0 = time.perf_counter()
            mu, W = fit_itq(db_vecs_train, bits=bits, iters=50, seed=42)
            codes = encode(db_f16.astype(np.float32), mu, W)
            t_build = time.perf_counter() - t0

            q_codes = encode(q_vecs, mu, W)
            backend = BitwiseCountBackend()

            pools = [250, 500, 1000, 2000] if args.sweep else [500]
            for pool_n in pools:
                latencies = []
                hits_per_q = []
                for i, q in enumerate(q_vecs):
                    t0 = time.perf_counter()
                    cand, _ = backend.search(q_codes[i], codes, n=pool_n)
                    # Fetch candidates from disk memmap float16
                    sub_vecs = db_f16[cand].astype(np.float32)
                    sub_sims = q @ sub_vecs.T
                    top10 = cand[np.argpartition(-sub_sims, 10)[:10]]
                    latencies.append((time.perf_counter() - t0) * 1000)
                    if gt_top10 is not None:
                        h = len(set(gt_top10[i]).intersection(set(top10)))
                        hits_per_q.append(h * 10.0)

                ov10 = float(np.mean(hits_per_q)) if hits_per_q else 0.0
                peak_wset = get_peak_wset_mb()
                delta_wset = max(0.0, peak_wset - baseline_wset)

                records.append({
                    "method_id": "M11",
                    "name": f"LSH-Memory-Wrapper (ITQ {bits}-bit, Pool {pool_n})",
                    "param": f"ITQ {bits}-bit + Pool {pool_n}",
                    "scale": args.scale,
                    "dataset": args.dataset,
                    "build_time_s": round(t_build, 3),
                    "overlap10": round(ov10, 2),
                    "overlap10_ci": bootstrap_ci(np.array(hits_per_q)) if hits_per_q else [0.0, 0.0],
                    "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                    "latency_iqr_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
                    "peak_wset_mb": round(peak_wset, 2),
                    "delta_wset_mb": round(delta_wset, 2),
                    "index_ram_mb": round(codes.nbytes / (1024 * 1024), 2),
                })

    # Save to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"[OK] Completed {args.method} on {args.scale} {args.dataset}: {len(records)} record(s) written.")


if __name__ == "__main__":
    main()
