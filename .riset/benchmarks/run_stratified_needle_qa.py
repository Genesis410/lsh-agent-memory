"""Task 4: Stratified Needle QA Benchmark across 5 Age Brackets and 2 Difficulty Levels.

Evaluates 7 baseline architectures:
- B1: No Memory (0%)
- B2: Sliding Window (10 turns)
- B3: Recent-50 Context Window + Exact Cosine
- B4: BM25 Lexical Retrieval (rank-bm25)
- B5: LSH-Memory-Wrapper alpha=0.7 (Recency Half-Life h=1000)
- B6: LSH-Memory-Wrapper alpha=1.0 (Pure Semantic Cosine)
- B7: Exact Cosine Upper Bound (In-Memory Float32)

Evaluated across 5 random seeds (150 needle trials per method per difficulty level)
with Wilson 95% Confidence Intervals.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rank_bm25 import BM25Okapi

from lsh_agent_memory.data.synthetic import (
    generate_stratified_needle_conversations,
    wilson_score_interval,
    AGE_BRACKETS,
    STRATIFIED_NEEDLES,
)
from lsh_agent_memory.data.extractor import SentenceExtractor
from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def episodic_score(cos_sim: float, delta_t: int, alpha: float = 0.7, half_life: float = 1000.0) -> float:
    """Episodic score combining cosine similarity and recency decay."""
    recency = 2.0 ** (-delta_t / half_life)
    return alpha * cos_sim + (1.0 - alpha) * recency


def run_needle_benchmark():
    print("=================================================================")
    print("TASK 4: STRATIFIED NEEDLE QA BENCHMARK (30 NEEDLES, 5 BRACKETS, 2 DIFFICULTIES)")
    print("=================================================================")

    extractor = SentenceExtractor()
    backend = BitwiseCountBackend()

    SEEDS = [42, 43, 44, 45, 46]
    METHODS = [
        "B1_No_Memory",
        "B2_Sliding_Window_10",
        "B3_Recent_50_Cosine",
        "B4_BM25_Lexical",
        "B5_LSH_Wrapper_Alpha_0.7",
        "B6_LSH_Wrapper_Alpha_1.0",
        "B7_Exact_Cosine_Upper_Bound",
    ]

    # Structure to hold raw hits: method -> difficulty -> bracket -> list of 0/1
    raw_results: Dict[str, Dict[str, Dict[str, List[int]]]] = {
        m: {
            "high_overlap": {b["name"]: [] for b in AGE_BRACKETS},
            "low_overlap": {b["name"]: [] for b in AGE_BRACKETS},
        }
        for m in METHODS
    }

    # Track MRRs
    raw_mrrs: Dict[str, Dict[str, List[float]]] = {
        m: {"high_overlap": [], "low_overlap": []} for m in METHODS
    }

    print(f"Running across {len(SEEDS)} random seeds with {len(STRATIFIED_NEEDLES)} needles each...")

    for seed_idx, seed in enumerate(SEEDS):
        t0_seed = time.perf_counter()
        turns, needles = generate_stratified_needle_conversations(total_turns=5000, seed=seed)
        total_turns = len(turns)
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{len(SEEDS)}): {total_turns} turns, {len(needles)} needles placed ---")

        # 1. Precompute turn embeddings
        all_texts = [t.text for t in turns]
        turn_embs = extractor.encode_texts(all_texts)
        turn_embs /= np.maximum(np.linalg.norm(turn_embs, axis=1, keepdims=True), 1e-12)

        # 2. Fit ITQ Hasher on turn embeddings (sample 2,500 turns)
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(total_turns, size=min(2500, total_turns), replace=False)
        mu_itq, W_itq = fit_itq(turn_embs[sample_indices], bits=128, iters=50, seed=seed)
        turn_codes = encode(turn_embs, mu_itq, W_itq)

        # 3. Setup BM25 corpus
        tokenized_corpus = [tokenize(t) for t in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        # 4. Evaluate each needle
        for needle_item in needles:
            needle_turn_id = needle_item["turn_id"]
            delta_t = needle_item["delta_t"]
            bracket_name = needle_item["bracket_name"]

            for diff_mode in ["high_overlap", "low_overlap"]:
                query_text = needle_item["query_high"] if diff_mode == "high_overlap" else needle_item["query_low"]
                q_vec = extractor.encode_texts([query_text])[0]
                q_vec /= max(np.linalg.norm(q_vec), 1e-12)
                q_code = encode(q_vec[None, :], mu_itq, W_itq)[0]

                # --- B1: No Memory ---
                hit_b1 = 0
                mrr_b1 = 0.0
                raw_results["B1_No_Memory"][diff_mode][bracket_name].append(hit_b1)
                raw_mrrs["B1_No_Memory"][diff_mode].append(mrr_b1)

                # --- B2: Sliding Window 10 turns ---
                # Window: [total_turns - 10, total_turns - 1]
                window_start = total_turns - 10
                hit_b2 = 1 if needle_turn_id >= window_start else 0
                rank_b2 = (total_turns - needle_turn_id) if hit_b2 else 0
                mrr_b2 = (1.0 / rank_b2) if (hit_b2 and rank_b2 <= 10) else 0.0
                raw_results["B2_Sliding_Window_10"][diff_mode][bracket_name].append(hit_b2)
                raw_mrrs["B2_Sliding_Window_10"][diff_mode].append(mrr_b2)

                # --- B3: Recent 50 Context Window + Exact Cosine ---
                r50_start = max(0, total_turns - 50)
                if needle_turn_id >= r50_start:
                    r50_indices = np.arange(r50_start, total_turns)
                    r50_sims = turn_embs[r50_indices] @ q_vec
                    r50_top10_local = np.argsort(-r50_sims)[:10]
                    r50_top10 = r50_indices[r50_top10_local]
                    hit_b3 = 1 if needle_turn_id in r50_top10 else 0
                    if hit_b3:
                        rank_b3 = list(r50_top10).index(needle_turn_id) + 1
                        mrr_b3 = 1.0 / rank_b3
                    else:
                        mrr_b3 = 0.0
                else:
                    hit_b3 = 0
                    mrr_b3 = 0.0
                raw_results["B3_Recent_50_Cosine"][diff_mode][bracket_name].append(hit_b3)
                raw_mrrs["B3_Recent_50_Cosine"][diff_mode].append(mrr_b3)

                # --- B4: BM25 Lexical Retrieval ---
                q_tokens = tokenize(query_text)
                bm25_scores = bm25.get_scores(q_tokens)
                bm25_top10 = np.argsort(-bm25_scores)[:10]
                hit_b4 = 1 if needle_turn_id in bm25_top10 else 0
                if hit_b4:
                    rank_b4 = list(bm25_top10).index(needle_turn_id) + 1
                    mrr_b4 = 1.0 / rank_b4
                else:
                    mrr_b4 = 0.0
                raw_results["B4_BM25_Lexical"][diff_mode][bracket_name].append(hit_b4)
                raw_mrrs["B4_BM25_Lexical"][diff_mode].append(mrr_b4)

                # --- Exact Cosine for Upper Bound & Base for Scorers ---
                all_sims = turn_embs @ q_vec
                exact_top10 = np.argsort(-all_sims)[:10]
                hit_b7 = 1 if needle_turn_id in exact_top10 else 0
                if hit_b7:
                    rank_b7 = list(exact_top10).index(needle_turn_id) + 1
                    mrr_b7 = 1.0 / rank_b7
                else:
                    mrr_b7 = 0.0
                raw_results["B7_Exact_Cosine_Upper_Bound"][diff_mode][bracket_name].append(hit_b7)
                raw_mrrs["B7_Exact_Cosine_Upper_Bound"][diff_mode].append(mrr_b7)

                # --- B6: LSH-Memory-Wrapper alpha=1.0 (Pure Semantic Cosine) ---
                # Hamming candidate pool: top 10% (500 candidates)
                pool_size = 500
                cand_indices, _ = backend.search(q_code, turn_codes, n=pool_size)
                cand_sims = turn_embs[cand_indices] @ q_vec
                lsh_top10_b6 = [cand_indices[i] for i in np.argsort(-cand_sims)[:10]]
                hit_b6 = 1 if needle_turn_id in lsh_top10_b6 else 0
                if hit_b6:
                    rank_b6 = lsh_top10_b6.index(needle_turn_id) + 1
                    mrr_b6 = 1.0 / rank_b6
                else:
                    mrr_b6 = 0.0
                raw_results["B6_LSH_Wrapper_Alpha_1.0"][diff_mode][bracket_name].append(hit_b6)
                raw_mrrs["B6_LSH_Wrapper_Alpha_1.0"][diff_mode].append(mrr_b6)

                # --- B5: LSH-Memory-Wrapper alpha=0.7 (Episodic Recency Decay h=1000) ---
                # Score candidates with alpha=0.7 and recency
                cand_dt = total_turns - np.array(cand_indices)
                cand_recency = 2.0 ** (-cand_dt / 1000.0)
                episodic_scores = 0.7 * cand_sims + 0.3 * cand_recency
                lsh_top10_b5 = [cand_indices[i] for i in np.argsort(-episodic_scores)[:10]]
                hit_b5 = 1 if needle_turn_id in lsh_top10_b5 else 0
                if hit_b5:
                    rank_b5 = lsh_top10_b5.index(needle_turn_id) + 1
                    mrr_b5 = 1.0 / rank_b5
                else:
                    mrr_b5 = 0.0
                raw_results["B5_LSH_Wrapper_Alpha_0.7"][diff_mode][bracket_name].append(hit_b5)
                raw_mrrs["B5_LSH_Wrapper_Alpha_0.7"][diff_mode].append(mrr_b5)

        elapsed_seed = time.perf_counter() - t0_seed
        print(f"Seed {seed} completed in {elapsed_seed:.1f}s.")

    # -------------------------------------------------------------
    # Aggregate and Compute Wilson Confidence Intervals
    # -------------------------------------------------------------
    print("\n--- Aggregating Results with Wilson 95% Confidence Intervals ---")
    summary: Dict[str, Any] = {}

    for method in METHODS:
        summary[method] = {
            "overall_high_overlap": {},
            "overall_low_overlap": {},
            "brackets_high_overlap": {},
            "brackets_low_overlap": {},
            "mrr_high_overlap": round(float(np.mean(raw_mrrs[method]["high_overlap"])), 4),
            "mrr_low_overlap": round(float(np.mean(raw_mrrs[method]["low_overlap"])), 4),
        }

        for diff in ["high_overlap", "low_overlap"]:
            all_hits = []
            for b in AGE_BRACKETS:
                b_name = b["name"]
                hits = raw_results[method][diff][b_name]
                all_hits.extend(hits)
                p, low, high = wilson_score_interval(sum(hits), len(hits))
                summary[method][f"brackets_{diff}"][b_name] = {
                    "hit_rate_pct": p,
                    "wilson_ci_95": [low, high],
                    "successes": sum(hits),
                    "total": len(hits),
                }

            p_all, low_all, high_all = wilson_score_interval(sum(all_hits), len(all_hits))
            summary[method][f"overall_{diff}"] = {
                "hit_rate_pct": p_all,
                "wilson_ci_95": [low_all, high_all],
                "successes": sum(all_hits),
                "total": len(all_hits),
            }

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Method':<28} | {'High Overlap Hit@10 (95% CI)':<30} | {'Low Overlap Paraphrase (95% CI)':<30}")
    print("-" * 90)
    for m in METHODS:
        h_info = summary[m]["overall_high_overlap"]
        l_info = summary[m]["overall_low_overlap"]
        h_str = f"{h_info['hit_rate_pct']}% [{h_info['wilson_ci_95'][0]}, {h_info['wilson_ci_95'][1]}]"
        l_str = f"{l_info['hit_rate_pct']}% [{l_info['wilson_ci_95'][0]}, {l_info['wilson_ci_95'][1]}]"
        print(f"{m:<28} | {h_str:<30} | {l_str:<30}")
    print("=" * 90)

    # Print Age Breakdown for Key Methods (BM25 vs LSH a=1.0 vs LSH a=0.7)
    print("\nAge Bracket Breakdown (Low-Overlap Paraphrased Queries):")
    print(f"{'Bracket':<14} | {'BM25':<16} | {'LSH (a=0.7)':<16} | {'LSH (a=1.0)':<16} | {'Exact Upper Bound':<16}")
    print("-" * 86)
    for b in AGE_BRACKETS:
        b_name = b["name"]
        bm25_p = summary["B4_BM25_Lexical"]["brackets_low_overlap"][b_name]["hit_rate_pct"]
        lsh7_p = summary["B5_LSH_Wrapper_Alpha_0.7"]["brackets_low_overlap"][b_name]["hit_rate_pct"]
        lsh1_p = summary["B6_LSH_Wrapper_Alpha_1.0"]["brackets_low_overlap"][b_name]["hit_rate_pct"]
        ex_p = summary["B7_Exact_Cosine_Upper_Bound"]["brackets_low_overlap"][b_name]["hit_rate_pct"]
        print(f"{b_name:<14} | {bm25_p:>5.1f}%          | {lsh7_p:>5.1f}%          | {lsh1_p:>5.1f}%          | {ex_p:>5.1f}%")
    print("-" * 86)

    # Save to file
    out_file = Path("benchmarks/stratified_needle_qa_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Task 4 results saved to {out_file}!")
    return summary


if __name__ == "__main__":
    run_needle_benchmark()
