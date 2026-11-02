"""Generate and cache standardized benchmark datasets: SV (Sintetis-Vektor) and SKR (Sintetis-Kovarians Riil).

Also precompute exact cosine ground truth for fair, deterministic evaluation.
"""

import os
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def generate_all_datasets():
    print("=================================================================")
    print("GENERATING STANDARDIZED BENCHMARK DATASETS (SV & SKR: 50k, 200k, 1M)")
    print("=================================================================")
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = 384

    # 1. Load empirical covariance and mean from Task 0
    cov_path = Path("data/pingpong/covariance_384.npz")
    assert cov_path.exists(), "Empirical covariance not found! Run diagnose_hash_quality.py first."
    cov_data = np.load(cov_path)
    mean_real = cov_data["mean"].astype(np.float32)
    cov_real = cov_data["cov"].astype(np.float32)

    # Eigen-decomposition for fast sampling: X = mu + Z @ (V * sqrt(S)).T
    w_cov, v_cov = np.linalg.eigh(cov_real)
    w_cov = np.maximum(w_cov, 1e-12)
    sqrt_cov = v_cov @ np.diag(np.sqrt(w_cov))

    rng = np.random.default_rng(42)

    # 2. Generate 1,000 evaluation queries (sampled from SKR distribution)
    q_path = out_dir / "queries_1000.npy"
    if not q_path.exists():
        print("Generating 1,000 SKR evaluation queries...")
        z_q = rng.standard_normal((1000, dim), dtype=np.float32)
        q_vecs = mean_real + z_q @ sqrt_cov.T
        q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
        np.save(q_path, q_vecs)
        print(f"Saved {q_path}")
    else:
        q_vecs = np.load(q_path)
    print(f"Loaded {len(q_vecs)} evaluation queries.")

    # 3. Generate SV and SKR datasets for scales: 50k, 200k, 1M
    scales = [50_000, 200_000, 1_000_000]

    for scale in scales:
        scale_label = f"{scale // 1000}k" if scale < 1_000_000 else "1m"
        print(f"\nProcessing scale: {scale_label} ({scale:,} vectors)...")

        # (A) Sintetis-Vektor (SV) - for Latency & Memory stress testing
        sv_path = out_dir / f"sv_{scale_label}.npy"
        if not sv_path.exists():
            print(f"  Generating SV {scale_label}...")
            n_clusters = max(10, scale // 100)
            c_centers = rng.standard_normal((n_clusters, dim), dtype=np.float32)
            c_centers /= np.linalg.norm(c_centers, axis=1, keepdims=True)

            # Generate via chunks to conserve RAM
            sv_mmap = np.lib.format.open_memmap(sv_path, mode="w+", dtype=np.float32, shape=(scale, dim))
            chunk_sz = 100_000
            for start in range(0, scale, chunk_sz):
                end = min(start + chunk_sz, scale)
                sz = end - start
                c_idx = rng.integers(0, n_clusters, size=sz)
                noise = rng.normal(0, 0.05, size=(sz, dim)).astype(np.float32)
                chunk_data = c_centers[c_idx] + noise
                chunk_data /= np.linalg.norm(chunk_data, axis=1, keepdims=True)
                sv_mmap[start:end] = chunk_data
            sv_mmap.flush()
            del sv_mmap
            print(f"  Saved {sv_path}")
        else:
            print(f"  SV {scale_label} already exists.")

        # (B) Sintetis-Kovarians Riil (SKR) - for Pareto Recall Curves
        skr_path = out_dir / f"skr_{scale_label}.npy"
        skr_f16_path = out_dir / f"skr_{scale_label}.f16.dat"

        if not skr_path.exists():
            print(f"  Generating SKR {scale_label}...")
            skr_mmap = np.lib.format.open_memmap(skr_path, mode="w+", dtype=np.float32, shape=(scale, dim))
            chunk_sz = 100_000
            for start in range(0, scale, chunk_sz):
                end = min(start + chunk_sz, scale)
                sz = end - start
                z = rng.standard_normal((sz, dim), dtype=np.float32)
                chunk_data = mean_real + z @ sqrt_cov.T
                chunk_data /= np.linalg.norm(chunk_data, axis=1, keepdims=True)
                skr_mmap[start:end] = chunk_data
            skr_mmap.flush()
            del skr_mmap
            print(f"  Saved {skr_path}")
        else:
            print(f"  SKR {scale_label} already exists.")

        # Create Float16 disk file for M2 and M11
        if not skr_f16_path.exists():
            print(f"  Exporting SKR {scale_label} to Float16 memmap dat...")
            skr_src = np.load(skr_path, mmap_mode="r")
            f16_mmap = np.memmap(skr_f16_path, dtype=np.float16, mode="w+", shape=(scale, dim))
            chunk_sz = 100_000
            for start in range(0, scale, chunk_sz):
                end = min(start + chunk_sz, scale)
                f16_mmap[start:end] = skr_src[start:end].astype(np.float16)
            f16_mmap.flush()
            del f16_mmap, skr_src
            print(f"  Saved {skr_f16_path}")

        # (C) Precompute Exact Ground Truth Top-10 on SKR
        gt_path = out_dir / f"gt_top10_skr_{scale_label}.npy"
        if not gt_path.exists():
            print(f"  Precomputing exact cosine ground truth top-10 for SKR {scale_label}...")
            t0 = time.perf_counter()
            skr_src = np.load(skr_path, mmap_mode="r")

            n_q = len(q_vecs)
            running_top_scores = np.full((n_q, 10), -1e9, dtype=np.float32)
            running_top_indices = np.zeros((n_q, 10), dtype=np.int64)

            chunk_sz = 50_000
            for start in range(0, scale, chunk_sz):
                end = min(start + chunk_sz, scale)
                chunk_data = skr_src[start:end]
                # (n_q, chunk_sz)
                sims_chunk = q_vecs @ chunk_data.T

                # Merge with running top 10
                comb_scores = np.hstack([running_top_scores, sims_chunk])
                comb_indices = np.hstack([running_top_indices, np.arange(start, end)[None, :].repeat(n_q, axis=0)])

                top_part = np.argpartition(-comb_scores, 10, axis=1)[:, :10]
                row_idx = np.arange(n_q)[:, None]
                sub_scores = comb_scores[row_idx, top_part]
                sub_order = np.argsort(-sub_scores, axis=1)

                running_top_scores = sub_scores[row_idx, sub_order]
                running_top_indices = comb_indices[row_idx, top_part][row_idx, sub_order]

            np.save(gt_path, running_top_indices)
            t_gt = time.perf_counter() - t0
            print(f"  Saved {gt_path} in {t_gt:.2f}s!")
        else:
            print(f"  Ground truth {gt_path} already exists.")

    print("\n[OK] All benchmark datasets and ground truth precomputed successfully!")


if __name__ == "__main__":
    generate_all_datasets()
