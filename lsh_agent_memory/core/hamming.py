"""Hamming distance search backend using native vectorized bitwise count."""

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Tuple
import numpy as np


class HammingBackend(Protocol):
    """Protocol for Hamming distance candidate retrieval."""

    def search(
        self,
        q_code: np.ndarray,
        codes: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-n nearest neighbors by Hamming distance.

        Args:
            q_code: Query bitcode (1D uint8 or uint64).
            codes: Database bitcodes (2D uint8 or uint64).
            n: Number of candidates to retrieve.

        Returns:
            Tuple of (indices, distances) sorted ascending by distance.
        """
        ...


class BitwiseCountBackend:
    """Hamming search using NumPy 2.0+ native np.bitwise_count on uint64 words."""

    def __init__(self, workers: int = 12, min_parallel_size: int = 250_000) -> None:
        """Initialize backend with thread pool and parallel threshold.

        Args:
            workers: Number of threads for chunked scanning.
            min_parallel_size: Minimum entry count to activate multi-threading.
        """
        self.workers = workers
        self.min_parallel_size = min_parallel_size
        self._pool = None

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        return self._pool

    @staticmethod
    def _compute_chunk_dist(q_u64: np.ndarray, codes_u64: np.ndarray) -> np.ndarray:
        """Compute Hamming distance for a uint64 contiguous matrix using unrolled 1D columns."""
        w = codes_u64.shape[1] if codes_u64.ndim > 1 else 1
        if w == 2:
            return (
                np.bitwise_count(codes_u64[:, 0] ^ q_u64[0]) +
                np.bitwise_count(codes_u64[:, 1] ^ q_u64[1])
            ).astype(np.uint16)
        elif w == 4:
            return (
                np.bitwise_count(codes_u64[:, 0] ^ q_u64[0]) +
                np.bitwise_count(codes_u64[:, 1] ^ q_u64[1]) +
                np.bitwise_count(codes_u64[:, 2] ^ q_u64[2]) +
                np.bitwise_count(codes_u64[:, 3] ^ q_u64[3])
            ).astype(np.uint16)
        elif w == 1:
            if codes_u64.ndim == 1:
                return np.bitwise_count(codes_u64 ^ q_u64[0]).astype(np.uint16)
            return np.bitwise_count(codes_u64[:, 0] ^ q_u64[0]).astype(np.uint16)
        else:
            diff = codes_u64 ^ q_u64
            counts = np.bitwise_count(diff)
            return counts.sum(axis=1, dtype=np.uint16)

    def search(
        self,
        q_code: np.ndarray,
        codes: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-n candidates by Hamming distance.

        Args:
            q_code: Query code (uint8 array of shape (bytes,) or uint64 of shape (w,)).
            codes: Array of shape (N, bytes) or (N, w).
            n: Number of nearest candidates.

        Returns:
            Tuple of (sorted_indices, sorted_distances).
        """
        total = len(codes)
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint16)

        # Ensure uint64 view for fast 64-bit word operations
        if codes.dtype != np.uint64:
            codes_u64 = np.ascontiguousarray(codes).view(np.uint64)
            q_u64 = np.ascontiguousarray(q_code).view(np.uint64)
        else:
            codes_u64 = codes
            q_u64 = q_code

        n_clamped = min(n, total)

        if total < self.min_parallel_size or self.workers <= 1:
            dist = self._compute_chunk_dist(q_u64, codes_u64)
        else:
            chunk_size = (total + self.workers - 1) // self.workers
            pool = self._get_pool()
            futures = [
                pool.submit(self._compute_chunk_dist, q_u64, codes_u64[start:min(start + chunk_size, total)])
                for start in range(0, total, chunk_size)
            ]
            dist = np.concatenate([f.result() for f in futures])

        # ponytail: linear scan with unrolled bitwise_count scales to ~2M. Upgrade path: Multi-Index Hashing (MIH) if N > 5M.
        if n_clamped == total:
            sort_order = np.argsort(dist)
            return sort_order, dist[sort_order]

        part_idx = np.argpartition(dist, n_clamped)[:n_clamped]
        sub_sort = np.argsort(dist[part_idx])
        top_indices = part_idx[sub_sort]
        return top_indices, dist[top_indices]
