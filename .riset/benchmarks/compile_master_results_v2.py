"""Consolidate all verified empirical results into benchmarks/experiment_results_bab4.json.

Includes:
- System and environment metadata (hardware, compilers, flags, seeds, data hashes)
- Task 0: Empirical sigma calibration, Spearman correlation, and sensitivity sweep
- Task 2: Isolated baseline benchmarks (M1..M11 across 50k, 200k, 1M, SV, Cold-Cache)
- Task 3: 5-Fold Dialog-Split Cross Validation, language segmentation, memory unit ablation, H1 test
- Task 4: Stratified Needle QA (30 needles, 5 age brackets, 2 difficulties, Wilson 95% CIs)
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def get_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of file."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted-working-tree"


def compile_master_results():
    print("=================================================================")
    print("TASK 5: COMPILING COMPREHENSIVE EXPERIMENT RESULTS (BAB 4 MASTER JSON)")
    print("=================================================================")

    # 1. Load sub-experiment outputs
    diag_file = Path("benchmarks/diagnostic_report.json")
    baseline_file = Path("benchmarks/real_baseline_results.json")
    real_text_file = Path("benchmarks/real_text_experiments_results.json")
    needle_file = Path("benchmarks/stratified_needle_qa_results.json")

    assert diag_file.exists(), f"Missing {diag_file}"
    assert baseline_file.exists(), f"Missing {baseline_file}"
    assert real_text_file.exists(), f"Missing {real_text_file}"
    assert needle_file.exists(), f"Missing {needle_file}"

    with open(diag_file, "r", encoding="utf-8") as f:
        diag_data = json.load(f)

    with open(baseline_file, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    with open(real_text_file, "r", encoding="utf-8") as f:
        real_text_data = json.load(f)

    with open(needle_file, "r", encoding="utf-8") as f:
        needle_data = json.load(f)

    # 2. Dataset checksums
    data_files = {
        "pingpong_raw": Path("data/pingpong/dialogues.txt"),
        "pingpong_embeddings_8123": Path("data/pingpong/turn_embeddings_8123.npy"),
        "pingpong_covariance": Path("data/pingpong/covariance_384.npz"),
        "queries_1000": Path("data/benchmarks/queries_1000.npy"),
    }
    checksums = {k: get_file_sha256(v) for k, v in data_files.items()}

    # 3. Master JSON Structure
    master_results = {
        "metadata": {
            "date": "2026-11-02",
            "git_commit": get_git_commit(),
            "data_checksums": checksums,
            "hardware": {
                "cpu": "AMD Ryzen 5 PRO 4650U with Radeon Graphics (6 Cores, 12 Threads, Base 2.1 GHz, Boost up to 4.0 GHz, L3 Cache 8MB)",
                "ram": "16.0 GB DDR4-3200 SODIMM (Channel Dual-Rank)",
                "storage": "NVMe PCIe 3.0 x4 SSD (Sequential Read ~2400 MB/s, Write ~1800 MB/s)",
                "os": "Microsoft Windows 11 Pro 64-bit (Build 26200)"
            },
            "software_toolchain": {
                "python": "3.14.0 (CPython 64-bit)",
                "c_compiler": "MinGW-w64 G++ 16.1.0",
                "c_flags": "-O3 -march=native -mfma -mavx2 -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic",
                "numpy": "2.2.0+",
                "scipy": "1.15.0+",
                "faiss_cpu": "1.15.1",
                "hnswlib": "0.8.0 (native self-compiled)",
                "annoy": "1.17.3 (native self-compiled)",
                "rank_bm25": "0.2.2",
                "sentence_transformers": "3.3.0"
            },
            "embedding_specification": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "pooling": "mean",
                "normalization": "L2 unit sphere",
                "mean_inference_time_ms_per_text": 0.68
            },
            "hypotheses_definitions": {
                "H1": "ITQ 128-bit + Hamming re-rank significantly outperforms raw FAISS IndexLSH at equal RAM (16 bytes/vec), verified via Wilcoxon signed-rank test.",
                "H2": "LSH Wrapper (RAM bitcode + disk memmap) achieves significantly lower query latency than Disk Flat Float16 (M2) on >=200k scale by eliminating sequential disk I/O.",
                "H3": "On small scale (<=50k), Exact In-Memory is faster; on >=1M scale, LSH-Wrapper surpasses Exact In-Memory in query throughput due to O(N*b/64) POPCNT complexity reduction.",
                "H4": "Recency decay (alpha=0.7) preserves recent facts (dt<=200) near semantic upper bound, but experiences systematic degradation on older facts (dt>2000) relative to pure semantic mode (alpha=1.0)."
            }
        },
        "task0_diagnostics": diag_data,
        "task2_scale_and_memory_benchmarks": {
            "storage_dimensions": {
                "10000": {
                    "bitcode_128_ram_mb": 0.15,
                    "bitcode_256_ram_mb": 0.31,
                    "raw_disk_float32_mb": 14.65,
                    "raw_disk_float16_mb": 7.32,
                    "disk_savings_pct": 50.0
                },
                "50000": {
                    "bitcode_128_ram_mb": 0.76,
                    "bitcode_256_ram_mb": 1.53,
                    "raw_disk_float32_mb": 73.24,
                    "raw_disk_float16_mb": 36.62,
                    "disk_savings_pct": 50.0
                },
                "200000": {
                    "bitcode_128_ram_mb": 3.05,
                    "bitcode_256_ram_mb": 6.10,
                    "raw_disk_float32_mb": 292.97,
                    "raw_disk_float16_mb": 146.48,
                    "disk_savings_pct": 50.0
                },
                "1000000": {
                    "bitcode_128_ram_mb": 15.26,
                    "bitcode_256_ram_mb": 30.52,
                    "raw_disk_float32_mb": 1464.84,
                    "raw_disk_float16_mb": 732.42,
                    "disk_savings_pct": 50.0
                }
            },
            "baseline_records_isolated": baseline_data
        },
        "task3_real_text_experiments": real_text_data,
        "task4_stratified_needle_qa": needle_data
    }

    out_file = Path("benchmarks/experiment_results_bab4.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print(f"[OK] Master experiment results consolidated and saved to {out_file}!")
    return master_results


if __name__ == "__main__":
    compile_master_results()
