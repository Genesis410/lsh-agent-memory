"""Script to diagnose hash quality: empirical sigma calibration from 8,123 PingPong turns, local vs global Spearman correlation, and sigma sweep."""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar

from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend
from lsh_agent_memory.data.pingpong_loader import load_pingpong_qa, extract_pingpong_turns
from lsh_agent_memory.data.extractor import SentenceExtractor


def run_hash_diagnostics():
    print("=================================================================")
    print("TASK 0: EMPIRICAL SIGMA CALIBRATION & HASH QUALITY DIAGNOSTICS")
    print("=================================================================")
    report = {}

    # 1. Load and extract all 8,123 PingPong turns
    raw_data = load_pingpong_qa()
    turns = extract_pingpong_turns(raw_data)
    print(f"Loaded PingPong QA: {len(turns)} clean conversational turns across 99 dialogues.")
    assert len(turns) == 8123, f"Expected 8123 turns, got {len(turns)}"

    turn_texts = [t["text"] for t in turns]
    dialog_ids = np.array([t["dialog_id"] for t in turns], dtype=np.int32)
    turn_nums = np.array([t["turn_num"] for t in turns], dtype=np.int32)

    # 2. Encode or load cached embeddings
    cache_path = Path("data/pingpong/turn_embeddings_8123.npy")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
    else:
        print("Encoding 8,123 turns using SentenceExtractor (all-MiniLM-L6-v2)...")
        extractor = SentenceExtractor()
        t0 = time.perf_counter()
        embeddings = extractor.encode_texts(turn_texts)
        t_enc = time.perf_counter() - t0
        print(f"Encoded {len(embeddings)} turns in {t_enc:.2f}s ({t_enc/len(embeddings)*1000:.2f} ms/text).")
        np.save(cache_path, embeddings)
        print(f"Saved embeddings cache to {cache_path}")

    # Ensure L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    n_turns, dim = embeddings.shape

    # 3. Estimate Empirical Covariance & Mean (SKR manifold generator)
    print("\nEstimating empirical mean and covariance from 8,123 turns...")
    mu_real = np.mean(embeddings, axis=0)
    centered = embeddings - mu_real
    cov_real = (centered.T @ centered) / (n_turns - 1)
    cov_path = Path("data/pingpong/covariance_384.npz")
    np.savez(cov_path, mean=mu_real, cov=cov_real)
    print(f"Saved empirical covariance matrix to {cov_path}")

    # 4. Measure Real Turn-to-Turn Nearest Neighbor Distribution
    # To prevent self-matching and dialog-local trivial overlap, sample 500 queries across folds
    # and exclude turns with the same dialog_id where |turn_nums[i] - turn_nums[j]| <= 2
    print("\nMeasuring true turn-to-turn nearest neighbor cosine distribution...")
    rng = np.random.default_rng(42)
    sample_q_idx = rng.choice(n_turns, size=500, replace=False)
    q_vecs = embeddings[sample_q_idx]

    all_top1 = []
    all_top5 = []
    all_top10 = []
    profile_top10 = np.zeros(10, dtype=np.float64)

    for idx, q_i in enumerate(sample_q_idx):
        sims = embeddings @ q_vecs[idx]
        # Mask out self and same-dialog adjacent turns (|turn_nums - q_turn_num| <= 2)
        same_diag = dialog_ids == dialog_ids[q_i]
        adj = np.abs(turn_nums - turn_nums[q_i]) <= 2
        mask = same_diag & adj
        sims[mask] = -1.0  # Exclude

        top_indices = np.argsort(-sims)[:10]
        top_cosines = sims[top_indices]

        all_top1.append(top_cosines[0])
        all_top5.append(top_cosines[4])
        all_top10.append(top_cosines[9])
        profile_top10 += top_cosines

    profile_top10 /= len(sample_q_idx)

    real_dist = {
        "mean_top1": float(np.mean(all_top1)),
        "mean_top5": float(np.mean(all_top5)),
        "mean_top10": float(np.mean(all_top10)),
        "profile_top1_to_top10": [round(float(c), 4) for c in profile_top10],
    }
    report["real_turn_nearest_neighbor_profile"] = real_dist
    print(f"  Mean Real Top-1 Cosine:  {real_dist['mean_top1']:.4f}")
    print(f"  Mean Real Top-5 Cosine:  {real_dist['mean_top5']:.4f}")
    print(f"  Mean Real Top-10 Cosine: {real_dist['mean_top10']:.4f}")
    print(f"  Top-1..Top-10 Profile:   {real_dist['profile_top1_to_top10']}")

    # 5. Empirically Calibrate Synthetic Cluster Spread Sigma
    print("\nEmpirically calibrating cluster spread sigma to match real profile...")

    def simulate_cluster_profile(sigma_val: float) -> np.ndarray:
        sim_rng = np.random.default_rng(42)
        n_c = 50
        pts_per_c = 100
        centers = sim_rng.standard_normal((n_c, dim)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        pts = centers.repeat(pts_per_c, axis=0) + sim_rng.normal(0, sigma_val, (n_c * pts_per_c, dim)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        # Sample 50 points from different clusters
        q_idx = sim_rng.choice(len(pts), size=50, replace=False)
        q = pts[q_idx]

        sim_profile = np.zeros(10, dtype=np.float64)
        for i_q, q_v in enumerate(q):
            s = pts @ q_v
            s[q_idx[i_q]] = -1.0  # exclude self
            top10 = np.sort(s)[-10:][::-1]
            sim_profile += top10
        return sim_profile / len(q)

    def objective(sigma_val: float) -> float:
        sim_prof = simulate_cluster_profile(sigma_val)
        return float(np.mean((sim_prof - profile_top10) ** 2))

    res_opt = minimize_scalar(objective, bounds=(0.01, 0.10), method="bounded")
    calibrated_sigma = float(res_opt.x)
    simulated_best_profile = simulate_cluster_profile(calibrated_sigma)

    report["calibrated_sigma"] = {
        "calibrated_sigma_value": round(calibrated_sigma, 4),
        "target_real_top10_mean": real_dist["mean_top10"],
        "simulated_top10_profile": [round(float(c), 4) for c in simulated_best_profile],
        "fit_mse": float(res_opt.fun),
    }
    print(f"  Calibrated Sigma: {calibrated_sigma:.4f} (MSE: {res_opt.fun:.6f})")
    print(f"  Simulated Profile: {report['calibrated_sigma']['simulated_top10_profile']}")

    # 6. Measure Global vs Local Spearman Correlation on Real 8,123 Embeddings
    print("\nMeasuring Global vs Local Spearman Correlation on Real Embeddings...")
    mu_itq, W_itq = fit_itq(embeddings[:2500], bits=128, iters=50, seed=42)
    codes_all = encode(embeddings, mu_itq, W_itq)
    backend = BitwiseCountBackend()

    global_rhos = []
    local_rhos = []

    # Evaluate on 200 random queries
    eval_q_idx = rng.choice(n_turns, size=200, replace=False)
    for q_i in eval_q_idx:
        q_vec = embeddings[q_i]
        q_code = codes_all[q_i]

        diff = np.bitwise_xor(codes_all, q_code)
        h_dist = np.bitwise_count(diff).sum(axis=1)
        cos = embeddings @ q_vec

        # Global Spearman (across all pairs)
        r_glob, _ = spearmanr(h_dist, -cos)
        if not np.isnan(r_glob):
            global_rhos.append(r_glob)

        # Local Spearman (among top 20 nearest neighbors in cosine)
        top20_idx = np.argsort(-cos)[1:21]  # exclude self
        r_loc, _ = spearmanr(h_dist[top20_idx], -cos[top20_idx])
        if not np.isnan(r_loc):
            local_rhos.append(r_loc)

    report["spearman_diagnostics"] = {
        "global_mean_rho": float(np.mean(global_rhos)),
        "local_top20_mean_rho": float(np.mean(local_rhos)),
    }
    print(f"  Global Spearman (all pairs [0, pi]): {report['spearman_diagnostics']['global_mean_rho']:.4f}")
    print(f"  Local Spearman (top 20 nearest):     {report['spearman_diagnostics']['local_top20_mean_rho']:.4f}")

    # 7. Sweep Sensitivitas Sigma (Boundary Condition Analysis)
    print("\nRunning Sigma Sensitivity Sweep...")
    sigma_candidates = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25]
    sweep_results = {}

    for s_val in sigma_candidates:
        # Generate 10k synthetic test dataset with sigma
        n_c = 100
        pts_per_c = 100
        sim_rng = np.random.default_rng(42)
        centers = sim_rng.standard_normal((n_c, dim)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        data_s = centers.repeat(pts_per_c, axis=0) + sim_rng.normal(0, s_val, (n_c * pts_per_c, dim)).astype(np.float32)
        data_s /= np.linalg.norm(data_s, axis=1, keepdims=True)

        q_s_idx = sim_rng.choice(len(data_s), size=100, replace=False)
        q_s = data_s[q_s_idx]

        # Fit ITQ 128-bit
        mu_s, W_s = fit_itq(data_s[:2500], bits=128, iters=50, seed=42)
        codes_s = encode(data_s, mu_s, W_s)
        q_codes_s = encode(q_s, mu_s, W_s)

        # Ground truth exact cosine
        gt_s_sims = q_s @ data_s.T
        for idx_row, row_idx in enumerate(q_s_idx):
            gt_s_sims[idx_row, row_idx] = -1.0  # exclude self
        gt_s_top10 = np.argsort(-gt_s_sims, axis=1)[:, :10]

        # Evaluate Overlap@10 with Pool 500
        hits_s = 0
        for i in range(len(q_s)):
            cand, _ = backend.search(q_codes_s[i], codes_s, n=500)
            cand = cand[cand != q_s_idx[i]]  # exclude self
            sub_s = q_s[i] @ data_s[cand].T
            ret = cand[np.argsort(-sub_s)[:10]]
            hits_s += len(set(gt_s_top10[i]).intersection(set(ret)))

        overlap_s = (hits_s / (len(q_s) * 10)) * 100
        sweep_results[str(s_val)] = round(overlap_s, 2)
        print(f"  Sigma = {s_val:4.2f} -> Overlap@10 = {overlap_s:6.2f}%")

    report["sigma_sensitivity_sweep"] = sweep_results

    # Save diagnostic report
    out_path = Path("benchmarks/diagnostic_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Diagnostic report written to {out_path}!")
    return report


if __name__ == "__main__":
    run_hash_diagnostics()
