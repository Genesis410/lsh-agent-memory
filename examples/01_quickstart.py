"""01_quickstart.py - Minimal 20-line episodic memory demo."""

import numpy as np
from lsh_agent_memory import LSHMemoryIndex, TurnIndexClock

def main():
    print("=== LSH-Agent-Memory Quickstart Demo ===")
    clock = TurnIndexClock(start_turn=0.0)

    # Initialize index: 384 dimensions (all-MiniLM-L6-v2), 128-bit hash codes
    index = LSHMemoryIndex(
        dim=384,
        bits=128,
        store_dir="./quickstart_store",
        alpha=0.8,       # 80% semantic relevance, 20% recency decay
        half_life=50.0,  # Relevance halves every 50 dialogue turns
        clock=clock,
    )

    rng = np.random.default_rng(42)
    # Simulate 10 episodic memory interactions
    embeddings = rng.standard_normal((10, 384)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    conversations = [
        "User suka kopi hitam tanpa gula di pagi hari",
        "User bekerja sebagai software engineer remote",
        "User sedang mengerjakan riset tentang locality-sensitive hashing",
        "Kemarin user berdiskusi tentang optimasi CPU SIMD x86",
        "User memesan tiket kereta api ke Bandung minggu depan",
        "User meminta rangkuman bab 4 tentang hasil pengujian empiris",
        "User membeli laptop Lenovo ThinkPad dengan prosesor Ryzen",
        "User lebih suka penjelasan teknis yang padat dan langsung ke poin",
        "User berencana mempublikasikan repositori ini ke GitHub",
        "User menanyakan cara konfigurasi lisensi open-source Apache 2.0",
    ]

    # Add memories incrementally
    for turn, text in enumerate(conversations):
        clock.current_turn = float(turn)
        index.add(embeddings[turn:turn+1], texts=[text])

    # Fit data-dependent ITQ projection
    index.fit(embeddings)
    print(f"[OK] Stored {index.store.count} memories. ITQ projection fitted!")

    # Advance clock to turn 15
    clock.current_turn = 15.0

    # Query with semantic embedding close to turn 0 (kopi)
    query_vec = embeddings[0] + rng.normal(0, 0.05, 384).astype(np.float32)
    results = index.query(query_vec, k=3)

    print("\n--- Top Retrieved Memories (Semantic + Recency) ---")
    for rank, res in enumerate(results, start=1):
        print(f"{rank}. [Score: {res['score']:.4f}] Turn {int(res['timestamp'])}: {res['text']}")

if __name__ == "__main__":
    main()
