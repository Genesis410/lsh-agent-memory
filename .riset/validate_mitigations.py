"""Verification and mitigation benchmark addressing all advisor feedback."""

import os
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np

from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend
from lsh_agent_memory.core.index import LSHMemoryIndex
from lsh_agent_memory.data.pingpong_loader import load_pingpong_qa
from lsh_agent_memory.data.extractor import SentenceExtractor


def get_process_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def test_1m_chunked_ram():
    """Verify that 1,000,000 entries do NOT blow up RAM if ingested in chunks."""
    print("\n--- [TEST 1] 1M Entries Streaming Chunk Ingest RAM Test ---")
    dim = 384
    bits = 128
    total_entries = 1_000_000
    chunk_size = 50_000
    tmp_dir = tempfile.mkdtemp()

    try:
        index = LSHMemoryIndex(dim=dim, bits=bits, store_dir=tmp_dir, pool_size=500)
        # Fit on small 5k sample
        rng = np.random.default_rng(42)
        fit_sample = rng.standard_normal((5_000, dim)).astype(np.float32)
        fit_sample /= np.linalg.norm(fit_sample, axis=1, keepdims=True)
        index.fit(fit_sample)

        base_mem = get_process_rss_mb()
        print(f"Base RSS before 1M stream: {base_mem:.1f} MB")

        # Ingest 1M in chunks of 50k
        t0 = time.perf_counter()
        for c_idx in range(total_entries // chunk_size):
            chunk = rng.standard_normal((chunk_size, dim)).astype(np.float32)
            chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
            index.add(chunk)
            if (c_idx + 1) % 5 == 0 or (c_idx + 1) == (total_entries // chunk_size):
                curr_mem = get_process_rss_mb()
                print(f"  Ingested {(c_idx + 1) * chunk_size:,} entries | RSS RAM: {curr_mem:.1f} MB | Bitcodes: {index.codes.nbytes / (1024*1024):.2f} MB")

        total_time = time.perf_counter() - t0
        final_mem = get_process_rss_mb()
        bitcode_mb = index.codes.nbytes / (1024 * 1024)
        disk_gb = Path(tmp_dir, "vectors.bin").stat().st_size / (1024 * 1024 * 1024)

        print(f"1M Stream Ingest Completed in {total_time:.2f} s")
        print(f"Final Peak Process RSS: {final_mem:.1f} MB (Target < 2,048 MB: {'PASS' if final_mem < 2048 else 'FAIL'})")
        print(f"Bitcode Index RAM size: {bitcode_mb:.2f} MB (Target < 16 MB: {'PASS' if bitcode_mb <= 16.0 else 'FAIL'})")
        print(f"Raw Vectors on Disk: {disk_gb:.2f} GB")
        return final_mem, bitcode_mb
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_pingpong_dedup_and_recall():
    """Test PingPong QA with deduplicated dialogues and pool size tuning for Recall >= 85%."""
    print("\n--- [TEST 2] PingPong QA Deduplicated Dialogues & Pool Size Tuning ---")
    raw_data = load_pingpong_qa()
    # Deduplicate unique dialogues
    unique_dialogs = []
    seen = set()
    for row in raw_data:
        d = row["dialog"].strip()
        if d not in seen:
            seen.add(d)
            unique_dialogs.append(d)

    print(f"Total QA questions: {len(raw_data)} | Unique dialogues: {len(unique_dialogs)}")

    extractor = SentenceExtractor()
    db_embeddings = extractor.encode_texts(unique_dialogs)
    q_embeddings = extractor.encode_texts([r["question"] for r in raw_data])

    # Ground truth: Exact cosine
    sims = q_embeddings @ db_embeddings.T
    gt_top10 = np.argsort(-sims, axis=1)[:, :10]
    gt_top5 = gt_top10[:, :5]

    # Ground truth Target: matching dialogue for each question
    dialog_to_idx = {d: idx for idx, d in enumerate(unique_dialogs)}
    target_dialog_idx = [dialog_to_idx[r["dialog"].strip()] for r in raw_data]

    # Fit ITQ 128-bit
    mu, W = fit_itq(db_embeddings, bits=128, iters=50, seed=42)
    db_codes = encode(db_embeddings, mu, W)
    q_codes = encode(q_embeddings, mu, W)

    backend = BitwiseCountBackend()

    # Test varying pool sizes
    for pool_n in [25, 50, 75, 99]:
        hits_overlap_10 = 0
        hits_overlap_5 = 0
        hit_target_10 = 0
        hit_target_5 = 0
        recip_ranks = []

        for i in range(len(raw_data)):
            cand, _ = backend.search(q_codes[i], db_codes, n=pool_n)
            # Re-rank candidates by exact cosine
            c_sims = q_embeddings[i] @ db_embeddings[cand].T
            top_ranked = cand[np.argsort(-c_sims)]

            # Definition A: Overlap with exact cosine top-k
            top10_cand = top_ranked[:10]
            top5_cand = top_ranked[:5]
            hits_overlap_10 += len(set(gt_top10[i]).intersection(set(top10_cand))) / 10.0
            hits_overlap_5 += len(set(gt_top5[i]).intersection(set(top5_cand))) / 5.0

            # Definition B: Hit@k (Target dialogue presence in top-k)
            t_idx = target_dialog_idx[i]
            if t_idx in top10_cand:
                hit_target_10 += 1
            if t_idx in top5_cand:
                hit_target_5 += 1

            # MRR for target dialogue
            rank = np.where(top_ranked == t_idx)[0]
            recip_ranks.append(1.0 / (rank[0] + 1) if len(rank) > 0 else 0.0)

        overlap10 = (hits_overlap_10 / len(raw_data)) * 100
        overlap5 = (hits_overlap_5 / len(raw_data)) * 100
        hit10 = (hit_target_10 / len(raw_data)) * 100
        hit5 = (hit_target_5 / len(raw_data)) * 100
        mrr = float(np.mean(recip_ranks))

        print(f"Pool N={pool_n:2d} | Overlap@10: {overlap10:5.1f}% | Overlap@5: {overlap5:5.1f}% | Hit@5: {hit5:5.1f}% | Hit@10: {hit10:5.1f}% | MRR: {mrr:.4f}")

    return


def test_latency_breakdown_and_exact_cosine():
    """Measure exact latency breakdown and compare with Brute-force Cosine at 50k entries."""
    print("\n--- [TEST 3] Latency Breakdown & Brute-force Cosine Comparison (50,000 entries) ---")
    dim = 384
    n = 50_000
    n_queries = 200
    tmp_dir = tempfile.mkdtemp()

    try:
        rng = np.random.default_rng(42)
        db = rng.standard_normal((n, dim)).astype(np.float32)
        db /= np.linalg.norm(db, axis=1, keepdims=True)
        queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        index = LSHMemoryIndex(dim=dim, bits=128, store_dir=tmp_dir, pool_size=500)
        index.fit(db[:5_000])
        index.add(db)

        # Baseline: Exact Cosine Brute-force Flat (NumPy dot product)
        t_exact_list = []
        for q in queries:
            t0 = time.perf_counter()
            sims = db @ q
            _ = np.argpartition(-sims, 10)[:10]
            t_exact_list.append((time.perf_counter() - t0) * 1000)

        # LSH Breakdown
        t_hamming_list = []
        t_slice_list = []
        t_rerank_list = []
        t_total_list = []

        q_codes = encode(queries, index.mu, index.W)

        for i, q in enumerate(queries):
            t_start = time.perf_counter()

            # Step 1: Hamming Scan + argpartition
            t0 = time.perf_counter()
            cand, _ = index.hamming.search(q_codes[i], index.codes, n=500)
            t_hamming = (time.perf_counter() - t0) * 1000

            # Step 2: Disk Memmap vector slice
            t1 = time.perf_counter()
            vecs = index.store.get_vectors(cand)
            t_slice = (time.perf_counter() - t1) * 1000

            # Step 3: Exact Cosine Re-rank on 500 candidates
            t2 = time.perf_counter()
            sub_sims = vecs @ q
            _ = cand[np.argpartition(-sub_sims, 10)[:10]]
            t_rerank = (time.perf_counter() - t2) * 1000

            t_total = (time.perf_counter() - t_start) * 1000

            t_hamming_list.append(t_hamming)
            t_slice_list.append(t_slice)
            t_rerank_list.append(t_rerank)
            t_total_list.append(t_total)

        print(f"Brute-force Flat Cosine Latency: p50 = {np.percentile(t_exact_list, 50):.2f} ms | p95 = {np.percentile(t_exact_list, 95):.2f} ms")
        print(f"LSH Breakdown (Pool N=500):")
        print(f"  - Hamming Scan (POPCNT uint64): p50 = {np.percentile(t_hamming_list, 50):.2f} ms")
        print(f"  - Memmap Array Slice:          p50 = {np.percentile(t_slice_list, 50):.2f} ms")
        print(f"  - Cosine Re-rank (500 cand):    p50 = {np.percentile(t_rerank_list, 50):.2f} ms")
        print(f"  - Total LSH Query Latency:      p50 = {np.percentile(t_total_list, 50):.2f} ms | p95 = {np.percentile(t_total_list, 95):.2f} ms")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_large_needle_sample():
    """Run downstream Needle QA with 30 needles and 95% Confidence Interval."""
    print("\n--- [TEST 4] Downstream Needle QA (N=30 needles across 5,000 turns) ---")
    from lsh_agent_memory.data.synthetic import generate_synthetic_conversations
    from lsh_agent_memory.core.clock import TurnIndexClock

    total_turns = 6_000
    needle_interval = 200  # 30 needles
    turns, needles = generate_synthetic_conversations(
        total_turns=total_turns,
        needle_intervals=needle_interval,
        seed=42,
    )
    texts = [t.text for t in turns]
    turn_ids = np.array([t.turn_id for t in turns], dtype=np.float64)

    extractor = SentenceExtractor()
    t_emb0 = time.perf_counter()
    all_embeddings = extractor.encode_texts(texts)
    t_emb = time.perf_counter() - t_emb0
    print(f"Extracted {len(texts):,} embeddings in {t_emb:.2f} s ({t_emb/len(texts)*1000:.2f} ms/text)")

    tmp_dir = tempfile.mkdtemp()
    try:
        clock = TurnIndexClock(start_turn=float(total_turns))
        lsh_index = LSHMemoryIndex(
            dim=384,
            bits=128,
            store_dir=tmp_dir,
            clock=clock,
            pool_size=500,
            alpha=1.0,  # Pure semantic
        )
        lsh_index.fit(all_embeddings[:5_000])
        lsh_index.add(all_embeddings, texts=texts, timestamps=turn_ids)

        hits_lsh = 0
        hits_exact = 0
        hits_window = 0
        total_n = len(needles)

        for item in needles:
            q_vec = extractor.encode_texts([item["query"]])[0]
            target = item["fact"]

            # LSH
            res = lsh_index.query(q_vec, k=10)
            res_texts = [r["text"] for r in res]
            if target in res_texts:
                hits_lsh += 1

            # Exact
            sims = all_embeddings @ q_vec
            top_exact = [texts[idx] for idx in np.argsort(-sims)[:10]]
            if target in top_exact:
                hits_exact += 1

            # Window
            if target in texts[-10:]:
                hits_window += 1

        acc_lsh = (hits_lsh / total_n) * 100
        acc_exact = (hits_exact / total_n) * 100
        acc_window = (hits_window / total_n) * 100

        # Wilson score 95% CI
        z = 1.96
        p = hits_lsh / total_n
        denom = 1 + z**2 / total_n
        center = (p + z**2 / (2 * total_n)) / denom
        half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * total_n)) / total_n)) / denom
        ci_low = max(0.0, (center - half) * 100)
        ci_high = min(100.0, (center + half) * 100)

        print(f"Total Needles Evaluated: {total_n}")
        print(f"Sliding Window (10 turns): {acc_window:.1f}% ({hits_window}/{total_n})")
        print(f"Exact Cosine (Upper Bound): {acc_exact:.1f}% ({hits_exact}/{total_n})")
        print(f"LSH-Memory-Wrapper (alpha=1.0): {acc_lsh:.1f}% ({hits_lsh}/{total_n}) [95% CI: {ci_low:.1f}% - {ci_high:.1f}%]")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_1m_chunked_ram()
    test_pingpong_dedup_and_recall()
    test_latency_breakdown_and_exact_cosine()
    test_large_needle_sample()
