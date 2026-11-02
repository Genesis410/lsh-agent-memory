"""Benchmark for Recall@k accuracy and ablation experiments."""

import time
import numpy as np

from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend


def compute_ground_truth_cosine(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute exact top-k nearest neighbor indices via brute-force cosine similarity."""
    # queries: (Q, d), database: (N, d)
    sims = queries @ database.T  # (Q, N)
    # top-k highest similarity
    top_indices = np.argsort(-sims, axis=1)[:, :k]
    return top_indices


def evaluate_recall(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
) -> float:
    """Calculate mean Recall@k across all queries."""
    recalls = []
    for ret, gt in zip(retrieved_indices, ground_truth_indices):
        gt_set = set(gt)
        hits = len(gt_set.intersection(set(ret)))
        recalls.append(hits / len(gt))
    return float(np.mean(recalls))


def run_accuracy_benchmark(
    n_database: int = 50_000,
    n_queries: int = 100,
    dim: int = 384,
    bits_list: list = [64, 128],
    pool_sizes: list = [50, 100, 250, 500, 1000],
    seed: int = 42,
) -> dict:
    """Run comprehensive accuracy benchmark comparing ITQ vs Random Projection."""
    print(f"\n=== BENCHMARK ACCURACY (N={n_database}, Q={n_queries}, Dim={dim}) ===")
    rng = np.random.default_rng(seed)

    # Generate synthetic clustered embeddings (anisotropic, typical of text embeddings)
    n_clusters = 50
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    cluster_assignments = rng.integers(0, n_clusters, size=n_database)
    noise = rng.normal(0, 0.3, size=(n_database, dim)).astype(np.float32)
    db = centers[cluster_assignments] + noise
    db /= np.linalg.norm(db, axis=1, keepdims=True)

    # Queries
    q_assignments = rng.integers(0, n_clusters, size=n_queries)
    q_noise = rng.normal(0, 0.3, size=(n_queries, dim)).astype(np.float32)
    queries = centers[q_assignments] + q_noise
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Exact Ground Truth Cosine
    print("Computing exact ground truth cosine...")
    t0 = time.perf_counter()
    gt_top10 = compute_ground_truth_cosine(queries, db, k=10)
    gt_time = (time.perf_counter() - t0) * 1000
    print(f"Ground truth computed in {gt_time:.2f} ms")

    backend = BitwiseCountBackend()
    results = {}

    for bits in bits_list:
        print(f"\n--- Code Length: {bits} bits ---")
        # 1. Random Projection (Baseline LSH)
        t_fit0 = time.perf_counter()
        W_rand = rng.standard_normal((dim, bits)).astype(np.float32)
        mu_zero = np.zeros(dim, dtype=np.float32)
        codes_rand = encode(db, mu_zero, W_rand)
        q_codes_rand = encode(queries, mu_zero, W_rand)

        # 2. ITQ Projection (Data-dependent)
        sample_size = min(5_000, n_database)
        mu_itq, W_itq = fit_itq(db[:sample_size], bits=bits, iters=50, seed=seed)
        t_fit_itq = (time.perf_counter() - t_fit0) * 1000

        t_enc0 = time.perf_counter()
        codes_itq = encode(db, mu_itq, W_itq)
        q_codes_itq = encode(queries, mu_itq, W_itq)
        t_encode = time.perf_counter() - t_enc0

        print(f"ITQ Fit (5k sample): {t_fit_itq:.2f} ms | Encode {n_database} entries: {t_encode:.3f} s")

        for pool_n in pool_sizes:
            # Evaluate ITQ
            retrieved_itq = []
            for i in range(n_queries):
                cand_idx, _ = backend.search(q_codes_itq[i], codes_itq, n=pool_n)
                # Exact cosine re-ranking on candidates
                c_sims = queries[i] @ db[cand_idx].T
                rerank_top10 = cand_idx[np.argsort(-c_sims)[:10]]
                retrieved_itq.append(rerank_top10)

            recall_itq = evaluate_recall(retrieved_itq, gt_top10)

            # Evaluate Random Projection
            retrieved_rand = []
            for i in range(n_queries):
                cand_idx, _ = backend.search(q_codes_rand[i], codes_rand, n=pool_n)
                c_sims = queries[i] @ db[cand_idx].T
                rerank_top10 = cand_idx[np.argsort(-c_sims)[:10]]
                retrieved_rand.append(rerank_top10)

            recall_rand = evaluate_recall(retrieved_rand, gt_top10)

            print(
                f"Pool N={pool_n:4d} | Random LSH Recall@10: {recall_rand*100:5.2f}% | "
                f"ITQ Recall@10: {recall_itq*100:5.2f}% | Gain: +{(recall_itq-recall_rand)*100:4.2f}%"
            )
            results[(bits, pool_n)] = {
                "recall_itq": recall_itq,
                "recall_rand": recall_rand,
                "encode_sec": t_encode,
            }

    return results


if __name__ == "__main__":
    run_accuracy_benchmark(n_database=20_000, n_queries=50, dim=384)
