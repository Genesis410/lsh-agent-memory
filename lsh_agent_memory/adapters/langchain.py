"""Thin LangChain VectorStore-compatible adapter for LSHMemoryIndex."""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
from lsh_agent_memory.core.index import LSHMemoryIndex


class LSHVectorStore:
    """LangChain-compatible VectorStore adapter for LSHMemoryIndex."""

    def __init__(
        self,
        index: LSHMemoryIndex,
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> None:
        """Initialize LangChain VectorStore adapter.

        Args:
            index: Underlying LSHMemoryIndex instance.
            embedding_fn: Callable that accepts list of strings and returns 2D numpy array.
        """
        self.index = index
        self.embedding_fn = embedding_fn

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Add texts to memory index via embedding function."""
        if self.embedding_fn is None:
            raise ValueError("embedding_fn is required to add raw text strings")

        vectors = self.embedding_fn(texts)
        timestamps = None
        if metadatas:
            timestamps = np.array(
                [m.get("timestamp", self.index.clock.now()) for m in metadatas],
                dtype=np.float64,
            )

        indices = self.index.add(vectors, texts=texts, timestamps=timestamps)
        return indices.tolist()

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search similar memories given a vector embedding."""
        vec = np.array(embedding, dtype=np.float32)
        return self.index.query(vec, k=k)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search similar memories given a query text."""
        if self.embedding_fn is None:
            raise ValueError("embedding_fn is required for text query search")
        q_vec = self.embedding_fn([query])[0]
        return self.similarity_search_by_vector(q_vec.tolist(), k=k)

    # ponytail: minimal duck-typing interface for LangChain. Upgrade path: inherit langchain_core.vectorstores.VectorStore when installed.
