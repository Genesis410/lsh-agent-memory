"""Disk-backed storage for raw vectors, timestamps, and texts using np.memmap."""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


class MemmapStore:
    """Disk-backed storage using memory-mapped arrays for raw vectors and timestamps."""

    def __init__(
        self,
        store_dir: Union[str, Path],
        dim: int = 384,
        dtype: np.dtype = np.float32,
        initial_capacity: int = 50_000,
    ) -> None:
        """Initialize or reopen a memmap store.

        Args:
            store_dir: Directory path on disk to store data files.
            dim: Embedding dimensionality.
            dtype: Floating point precision for embeddings (float32 or float16).
            initial_capacity: Initial pre-allocated vector slots.
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.dtype = np.dtype(dtype)
        self.capacity = initial_capacity
        self.count = 0

        self.vec_file = self.store_dir / "vectors.bin"
        self.ts_file = self.store_dir / "timestamps.bin"
        self.meta_file = self.store_dir / "meta.json"
        self.texts_file = self.store_dir / "texts.jsonl"

        self._init_or_load()

    def _init_or_load(self) -> None:
        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.count = meta["count"]
            self.dim = meta["dim"]
            self.dtype = np.dtype(meta["dtype"])
            self.capacity = meta["capacity"]
            mode = "r+"
        else:
            mode = "w+"

        self.vectors = np.memmap(
            self.vec_file,
            dtype=self.dtype,
            mode=mode,
            shape=(self.capacity, self.dim),
        )
        self.timestamps = np.memmap(
            self.ts_file,
            dtype=np.float64,
            mode=mode,
            shape=(self.capacity,),
        )
        if mode == "w+":
            self._save_meta()

    def _save_meta(self) -> None:
        meta = {
            "count": self.count,
            "dim": self.dim,
            "dtype": self.dtype.name,
            "capacity": self.capacity,
        }
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _resize(self, needed_capacity: int) -> None:
        """Expand memory mapped file capacity."""
        # Flush current memmaps
        self.vectors.flush()
        self.timestamps.flush()
        del self.vectors
        del self.timestamps

        new_capacity = max(needed_capacity, self.capacity * 2)

        # Reopen with new capacity
        self.vectors = np.memmap(
            self.vec_file,
            dtype=self.dtype,
            mode="r+",
            shape=(new_capacity, self.dim),
        )
        self.timestamps = np.memmap(
            self.ts_file,
            dtype=np.float64,
            mode="r+",
            shape=(new_capacity,),
        )
        self.capacity = new_capacity
        self._save_meta()

    def append(
        self,
        vectors: np.ndarray,
        timestamps: np.ndarray,
        texts: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Append vectors, timestamps, and texts to store.

        Args:
            vectors: Embeddings of shape (M, dim).
            timestamps: Array of shape (M,).
            texts: Optional list of M text strings.

        Returns:
            Assigned indices of newly appended items.
        """
        m = len(vectors)
        if m == 0:
            return np.empty(0, dtype=np.int64)

        if self.count + m > self.capacity:
            self._resize(self.count + m)

        start = self.count
        end = start + m
        self.vectors[start:end] = vectors.astype(self.dtype)
        self.timestamps[start:end] = timestamps
        self.count = end
        self._save_meta()

        if texts is not None:
            with open(self.texts_file, "a", encoding="utf-8") as f:
                for t in texts:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

        # ponytail: texts stored in jsonl for simple streaming. Upgrade path: indexed binary offset table for O(1) text seek.
        return np.arange(start, end, dtype=np.int64)

    def get_vectors(self, indices: np.ndarray) -> np.ndarray:
        """Retrieve vector slice by indices."""
        return np.asarray(self.vectors[indices])

    def get_timestamps(self, indices: np.ndarray) -> np.ndarray:
        """Retrieve timestamp slice by indices."""
        return np.asarray(self.timestamps[indices])

    def get_texts(self, indices: np.ndarray) -> List[str]:
        """Read specific texts from jsonl file."""
        if not self.texts_file.exists():
            return ["" for _ in indices]

        needed_set = set(indices.tolist())
        results = {}
        with open(self.texts_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in needed_set:
                    results[idx] = json.loads(line)
                if len(results) == len(needed_set):
                    break

        return [results.get(i, "") for i in indices]

    def flush(self) -> None:
        """Flush changes to physical disk."""
        self.vectors.flush()
        self.timestamps.flush()
        self._save_meta()
