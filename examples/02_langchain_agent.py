"""02_langchain_agent.py - LangChain integration example."""

import numpy as np
from lsh_agent_memory import LSHMemoryIndex
from lsh_agent_memory.adapters import LSHVectorStore

def mock_embedding_fn(texts):
    """Simulated deterministic embedding generator (e.g. replacing sentence-transformers)."""
    rng = np.random.default_rng(len(texts))
    vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def main():
    print("=== LSH-Agent-Memory LangChain Adapter Demo ===")

    # 1. Initialize core index
    core_index = LSHMemoryIndex(
        dim=384,
        bits=128,
        store_dir="./langchain_store",
        alpha=0.85,
    )

    # 2. Wrap with LangChain-compatible VectorStore
    vectorstore = LSHVectorStore(
        index=core_index,
        embedding_fn=mock_embedding_fn,
    )

    # 3. Ingest documents/memories via LangChain API
    docs = [
        "User prefers responses formatted in concise technical markdown.",
        "System requirement: Host server has 16GB RAM and Ryzen 5 CPU.",
        "Episodic memory uses 128-bit ITQ hashing to preserve RAM.",
        "User schedule: Standup meeting every Monday at 09:00 WIB.",
    ]
    metadatas = [{"timestamp": float(i * 10)} for i in range(len(docs))]

    vectorstore.add_texts(docs, metadatas=metadatas)
    print(f"[OK] Added {len(docs)} documents to LSHVectorStore.")

    # 4. Perform similarity search via LangChain API
    query = "How is memory saved in this system?"
    results = vectorstore.similarity_search(query, k=2)

    print(f"\nQuery: '{query}'")
    print("--- LangChain Similarity Search Results ---")
    for r in results:
        print(f"- [Score: {r['score']:.4f}] {r['text']}")

if __name__ == "__main__":
    main()
