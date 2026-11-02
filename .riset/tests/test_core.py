"""Self-contained runnable assert-based check for LSH core components."""

import shutil
import tempfile
import numpy as np

from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend
from lsh_agent_memory.core.clock import TurnIndexClock
from lsh_agent_memory.core.scorer import EpisodicScorer
from lsh_agent_memory.core.store import MemmapStore
from lsh_agent_memory.core.index import LSHMemoryIndex


def test_hasher_and_encode():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 384)).astype(np.float32)
    mu, W = fit_itq(X, bits=64, iters=10, seed=42)

    assert mu.shape == (384,)
    assert W.shape == (384, 64)

    codes = encode(X, mu, W)
    assert codes.shape == (100, 8)  # 64 bits = 8 bytes
    assert codes.dtype == np.uint8


def test_hamming_backend():
    backend = BitwiseCountBackend(workers=2, min_parallel_size=10)
    rng = np.random.default_rng(42)
    # 50 codes of 128-bit (16 bytes = 2 uint64 words)
    codes = rng.integers(0, 256, size=(50, 16), dtype=np.uint8)
    q = codes[5].copy()  # Query exact match to index 5

    idx, dist = backend.search(q, codes, n=5)
    assert len(idx) == 5
    assert idx[0] == 5
    assert dist[0] == 0  # Distance to itself must be 0


def test_store_and_memmap():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = MemmapStore(store_dir=tmp_dir, dim=4, initial_capacity=10)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        ts = np.array([10.0, 20.0])
        texts = ["halo dunia", "hello world"]

        idx = store.append(vecs, ts, texts)
        assert np.array_equal(idx, [0, 1])
        assert store.count == 2

        ret_vecs = store.get_vectors(np.array([0, 1]))
        assert np.allclose(ret_vecs, vecs)

        ret_texts = store.get_texts(np.array([1, 0]))
        assert ret_texts == ["hello world", "halo dunia"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_episodic_scorer():
    clock = TurnIndexClock(start_turn=100.0)
    scorer = EpisodicScorer(alpha=0.5, half_life=50.0, clock=clock)

    q = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    # candidate 0 is at turn 100 (delta=0), candidate 1 is at turn 50 (delta=50 = 1 half-life)
    ts = np.array([100.0, 50.0])

    scores = scorer.score_candidates(q, candidates, ts)
    # cand 0: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    # cand 1: 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.75)


def test_index_end_to_end():
    tmp_dir = tempfile.mkdtemp()
    try:
        clock = TurnIndexClock(start_turn=0.0)
        index = LSHMemoryIndex(
            dim=64,
            bits=64,
            store_dir=tmp_dir,
            clock=clock,
            pool_size=10,
        )

        rng = np.random.default_rng(123)
        sample = rng.standard_normal((30, 64)).astype(np.float32)
        # Normalize
        sample /= np.linalg.norm(sample, axis=1, keepdims=True)

        texts = [f"turn_{i} memory text" for i in range(30)]
        index.add(sample, texts=texts)

        # Fit ITQ
        index.fit(sample[:20])
        assert index.is_fitted
        assert index.codes.shape == (30, 8)

        # Query
        res = index.query(sample[5], k=3)
        assert len(res) == 3
        assert res[0]["index"] == 5
        assert res[0]["text"] == "turn_5 memory text"

        # Test reload
        index_reloaded = LSHMemoryIndex(
            dim=64,
            bits=64,
            store_dir=tmp_dir,
            clock=clock,
        )
        assert index_reloaded.is_fitted
        assert index_reloaded.store.count == 30
        res2 = index_reloaded.query(sample[5], k=1)
        assert res2[0]["index"] == 5
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_hasher_and_encode()
    print("test_hasher_and_encode: PASS")
    test_hamming_backend()
    print("test_hamming_backend: PASS")
    test_store_and_memmap()
    print("test_store_and_memmap: PASS")
    test_episodic_scorer()
    print("test_episodic_scorer: PASS")
    test_index_end_to_end()
    print("test_index_end_to_end: PASS")
    print("ALL CORE UNIT CHECKS PASSED.")
