"""Main LSH Episodic Memory Index orchestrator."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from lsh_agent_memory.core.clock import Clock, TurnIndexClock
from lsh_agent_memory.core.hasher import encode, fit_itq
from lsh_agent_memory.core.hamming import BitwiseCountBackend, HammingBackend
from lsh_agent_memory.core.scorer import EpisodicScorer
from lsh_agent_memory.core.store import MemmapStore


class LSHMemoryIndex:
    """Episodic memory index with data-dependent LSH, memmap store, and recency decay."""

    def __init__(
        self,
        dim: int = 384,
        bits: int = 128,
        store_dir: Union[str, Path] = "./mem_store",
        raw_dtype: np.dtype = np.float32,
        alpha: float = 0.8,
        half_life: float = 200.0,
        clock: Optional[Clock] = None,
        pool_size: int = 500,
        cold_start_threshold: int = 5_000,
        hamming_backend: Optional[HammingBackend] = None,
    ) -> None:
        """Initialize LSHMemoryIndex.

        Args:
            dim: Embedding dimension.
            bits: Bitcode length (e.g. 64 or 128).
            store_dir: Filesystem directory for memmap store.
            raw_dtype: Precision for raw vectors (float32 or float16).
            alpha: Relevance weighting (1.0 = pure semantic).
            half_life: Recency half-life.
            clock: Time clock implementation.
            pool_size: Candidate count N retrieved by Hamming search.
            cold_start_threshold: Minimum vector count to trigger automatic ITQ fit.
            hamming_backend: Custom Hamming search backend.
        """
        self.dim = dim
        self.bits = bits
        self.store_dir = Path(store_dir)
        self.raw_dtype = np.dtype(raw_dtype)
        self.pool_size = pool_size
        self.cold_start_threshold = cold_start_threshold

        self.clock = clock if clock is not None else TurnIndexClock()
        self.scorer = EpisodicScorer(alpha=alpha, half_life=half_life, clock=self.clock)
        self.store = MemmapStore(
            store_dir=self.store_dir,
            dim=self.dim,
            dtype=self.raw_dtype,
        )
        self.hamming = hamming_backend or BitwiseCountBackend()

        self.mu: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.codes: Optional[np.ndarray] = None  # (N, bits // 8) uint8 in RAM
        self.is_fitted = False

        self._check_and_load_existing_index()

    def _check_and_load_existing_index(self) -> None:
        cfg_file = self.store_dir / "index_config.json"
        mu_file = self.store_dir / "mu.npy"
        w_file = self.store_dir / "W.npy"
        codes_file = self.store_dir / "codes.npy"

        if cfg_file.exists() and mu_file.exists() and w_file.exists():
            self.mu = np.load(mu_file)
            self.W = np.load(w_file)
            if codes_file.exists():
                self.codes = np.load(codes_file)
            self.is_fitted = True

    def fit(self, sample_vectors: np.ndarray, iters: int = 50, seed: int = 0) -> None:
        """Train PCA + ITQ projection matrix on sample vectors.

        Args:
            sample_vectors: Representative embeddings of shape (S, dim).
            iters: Number of ITQ rotation iterations.
            seed: RNG seed for reproducible projection.
        """
        self.mu, self.W = fit_itq(sample_vectors, bits=self.bits, iters=iters, seed=seed)
        self.is_fitted = True

        # Re-encode all existing vectors in store if any
        if self.store.count > 0:
            existing = self.store.get_vectors(np.arange(self.store.count))
            self.codes = encode(existing, self.mu, self.W)
        else:
            self.codes = np.empty((0, self.bits // 8), dtype=np.uint8)

        self.save_index_state()

    def save_index_state(self) -> None:
        """Save projection matrix, centering vector, and RAM bitcodes."""
        if not self.is_fitted:
            return
        np.save(self.store_dir / "mu.npy", self.mu)
        np.save(self.store_dir / "W.npy", self.W)
        if self.codes is not None:
            np.save(self.store_dir / "codes.npy", self.codes)
        cfg = {
            "dim": self.dim,
            "bits": self.bits,
            "pool_size": self.pool_size,
            "is_fitted": self.is_fitted,
        }
        with open(self.store_dir / "index_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    def add(
        self,
        vectors: np.ndarray,
        texts: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Add new vectors and optional metadata to the index.

        Args:
            vectors: Embeddings of shape (M, dim).
            texts: Optional text strings of length M.
            timestamps: Optional array of timestamps. If None, uses clock.now().

        Returns:
            Assigned indices of newly added items.
        """
        vectors = np.asarray(vectors, dtype=self.raw_dtype)
        m = len(vectors)
        if m == 0:
            return np.empty(0, dtype=np.int64)

        if timestamps is None:
            now_val = self.clock.now()
            timestamps = np.full(m, now_val, dtype=np.float64)
        else:
            timestamps = np.asarray(timestamps, dtype=np.float64)

        # Append to persistent memmap store
        indices = self.store.append(vectors, timestamps, texts)

        # Update bitcodes
        if self.is_fitted:
            new_codes = encode(vectors, self.mu, self.W)
            if self.codes is None or len(self.codes) == 0:
                self.codes = new_codes
            else:
                # ponytail: in-memory codes array using np.concatenate on add. Upgrade path: pre-allocated code buffer with geometric growth.
                self.codes = np.concatenate([self.codes, new_codes], axis=0)
        elif self.store.count >= self.cold_start_threshold:
            # Auto-fit upon reaching threshold
            sample = self.store.get_vectors(np.arange(self.store.count))
            self.fit(sample)

        return indices

    def query(
        self,
        q_vec: np.ndarray,
        k: int = 10,
        pool_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search top-k relevant episodic memories.

        Args:
            q_vec: Query vector of shape (dim,).
            k: Number of final results.
            pool_n: Candidate pool size for Hamming filtering.

        Returns:
            List of result dicts containing id, score, text, and timestamp.
        """
        total = self.store.count
        if total == 0:
            return []

        q_vec = np.asarray(q_vec, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Cold start fallback: Brute-force exact cosine if not yet fitted
        if not self.is_fitted or self.codes is None or len(self.codes) < total:
            all_indices = np.arange(total)
            raw_vecs = self.store.get_vectors(all_indices).astype(np.float32)
            ts = self.store.get_timestamps(all_indices)
            scores = self.scorer.score_candidates(q_vec, raw_vecs, ts)
            k_clamped = min(k, total)
            top_k_idx = np.argsort(-scores)[:k_clamped]
            sel_idx = all_indices[top_k_idx]
            sel_scores = scores[top_k_idx]
            sel_texts = self.store.get_texts(sel_idx)
            sel_ts = ts[top_k_idx]

            return [
                {
                    "index": int(sel_idx[i]),
                    "score": float(sel_scores[i]),
                    "text": sel_texts[i],
                    "timestamp": float(sel_ts[i]),
                }
                for i in range(k_clamped)
            ]

        # Stage 1: Hamming distance filtering
        n_candidates = pool_n or self.pool_size
        n_candidates = min(n_candidates, total)

        q_code = encode(q_vec[np.newaxis, :], self.mu, self.W)[0]
        cand_indices, _ = self.hamming.search(q_code, self.codes[:total], n=n_candidates)

        if len(cand_indices) == 0:
            return []

        # Stage 2: Exact Cosine Re-ranking + Recency Decay
        cand_vecs = self.store.get_vectors(cand_indices).astype(np.float32)
        cand_ts = self.store.get_timestamps(cand_indices)
        scores = self.scorer.score_candidates(q_vec, cand_vecs, cand_ts)

        k_clamped = min(k, len(cand_indices))
        rerank_sort = np.argsort(-scores)[:k_clamped]

        final_indices = cand_indices[rerank_sort]
        final_scores = scores[rerank_sort]
        final_texts = self.store.get_texts(final_indices)
        final_ts = cand_ts[rerank_sort]

        return [
            {
                "index": int(final_indices[i]),
                "score": float(final_scores[i]),
                "text": final_texts[i],
                "timestamp": float(final_ts[i]),
            }
            for i in range(k_clamped)
        ]
