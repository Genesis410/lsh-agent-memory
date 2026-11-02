"""Downstream Needle-in-Haystack evaluation across episodic memory conditions."""

import shutil
import tempfile
import time
from typing import Dict, List
import numpy as np

from lsh_agent_memory.core.clock import TurnIndexClock
from lsh_agent_memory.core.index import LSHMemoryIndex
from lsh_agent_memory.data.synthetic import generate_synthetic_conversations
from lsh_agent_memory.data.extractor import SentenceExtractor


def run_downstream_needle_evaluation(
    total_turns: int = 5_000,
    needle_interval: int = 1_000,
    k_memories: int = 5,
    dim: int = 384,
) -> Dict[str, dict]:
    """Evaluate Needle retrieval accuracy across memory configurations."""
    print(f"\n=== DOWNSTREAM NEEDLE RETRIEVAL EVALUATION (Total Turns={total_turns:,}, Needles Every {needle_interval:,} Turns) ===")

    # 1. Generate Synthetic Data
    turns, needles = generate_synthetic_conversations(
        total_turns=total_turns,
        needle_intervals=needle_interval,
        seed=42,
    )
    texts = [t.text for t in turns]
    turn_ids = np.array([t.turn_id for t in turns], dtype=np.float64)

    # 2. Extract Embeddings
    print("Generating sentence embeddings...")
    extractor = SentenceExtractor(dim=dim)
    all_embeddings = extractor.encode_texts(texts)

    tmp_dir = tempfile.mkdtemp()
    try:
        clock = TurnIndexClock(start_turn=float(total_turns))
        lsh_index = LSHMemoryIndex(
            dim=dim,
            bits=128,
            store_dir=tmp_dir,
            clock=clock,
            pool_size=250,
            alpha=0.7,
            half_life=1000.0,
        )

        # Ingest and fit
        sample_fit = all_embeddings[:min(5_000, total_turns)]
        lsh_index.fit(sample_fit)
        lsh_index.add(all_embeddings, texts=texts, timestamps=turn_ids)

        conditions = ["no_memory", "sliding_window_10", "exact_cosine", "lsh_wrapper"]
        results = {c: {"hits": 0, "total": len(needles), "latencies_ms": []} for c in conditions}

        for n_item in needles:
            q_text = n_item["query"]
            q_vec = extractor.encode_texts([q_text])[0]
            target_fact = n_item["fact"]

            # Condition 1: No Memory
            # (No retrieval context injected)
            results["no_memory"]["latencies_ms"].append(0.0)

            # Condition 2: Sliding Window (last 10 turns)
            t0 = time.perf_counter()
            window_texts = texts[-10:]
            results["sliding_window_10"]["latencies_ms"].append((time.perf_counter() - t0) * 1000)
            if target_fact in window_texts:
                results["sliding_window_10"]["hits"] += 1

            # Condition 3: Exact Cosine (Upper Bound)
            t0 = time.perf_counter()
            sims = all_embeddings @ q_vec
            top_exact_idx = np.argsort(-sims)[:k_memories]
            results["exact_cosine"]["latencies_ms"].append((time.perf_counter() - t0) * 1000)
            retrieved_exact = [texts[idx] for idx in top_exact_idx]
            if target_fact in retrieved_exact:
                results["exact_cosine"]["hits"] += 1

            # Condition 4: LSH-Memory-Wrapper
            t0 = time.perf_counter()
            lsh_res = lsh_index.query(q_vec, k=k_memories)
            results["lsh_wrapper"]["latencies_ms"].append((time.perf_counter() - t0) * 1000)
            retrieved_lsh = [r["text"] for r in lsh_res]
            if target_fact in retrieved_lsh:
                results["lsh_wrapper"]["hits"] += 1

        print("\n--- Downstream Needle Retrieval Results ---")
        for cond, d in results.items():
            acc = (d["hits"] / d["total"]) * 100
            mean_lat = float(np.mean(d["latencies_ms"]))
            print(f"Condition: {cond:<18} | Accuracy: {acc:5.1f}% ({d['hits']}/{d['total']}) | Mean Latency: {mean_lat:6.2f} ms")

        return results

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_downstream_needle_evaluation(total_turns=4_000, needle_interval=800)
