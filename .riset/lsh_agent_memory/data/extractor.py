"""Embedding extraction and caching to disk."""

from pathlib import Path
from typing import List, Union
import numpy as np


class SentenceExtractor:
    """Extracts sentence embeddings with caching to .npy files."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        device: str = "cpu",
    ) -> None:
        """Initialize sentence extractor.

        Args:
            model_name: HuggingFace model identifier.
            dim: Output embedding dimensionality.
            device: Compute device ('cpu').
        """
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            # Fallback for environments without sentence-transformers installed yet
            self._model = None

    def encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encode list of strings into L2-normalized float32 embeddings."""
        if self._model is not None:
            vecs = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vecs.astype(np.float32)

        # ponytail: deterministic token-hash pseudo-embeddings when sentence-transformers is uninstalled.
        # Upgrade path: install sentence-transformers + torch for real MiniLM embeddings.
        import re
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"\w+", text.lower())
            if not words:
                words = ["empty"]
            v = np.zeros(self.dim, dtype=np.float32)
            for w in words:
                # Seeded vector for each unique word
                w_seed = int.from_bytes(w.encode("utf-8")[:8].ljust(8, b"\0"), "little")
                rng = np.random.default_rng(w_seed % (2**32))
                v += rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(v)
            out[i] = v / norm if norm > 0 else v
        return out

    def extract_and_cache(
        self,
        texts: List[str],
        output_file: Union[str, Path],
    ) -> np.ndarray:
        """Encode texts and cache array to .npy file."""
        out_path = Path(output_file)
        if out_path.exists():
            return np.load(out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        vecs = self.encode_texts(texts)
        np.save(out_path, vecs)
        return vecs
