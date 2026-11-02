"""03_llamacpp_injector.py - Prompt context injection for local LLMs."""

import numpy as np
from lsh_agent_memory import LSHMemoryIndex, TurnIndexClock
from lsh_agent_memory.adapters import LlamaCppMemoryInjector

def main():
    print("=== LSH-Agent-Memory Llama.cpp Prompt Injector Demo ===")

    clock = TurnIndexClock(start_turn=0.0)
    index = LSHMemoryIndex(
        dim=384,
        bits=128,
        store_dir="./llamacpp_store",
        clock=clock,
    )

    rng = np.random.default_rng(101)

    # Store past dialogue interactions
    interactions = [
        "Nama saya Budi dan saya berdomisili di Jakarta Selatan.",
        "Saya memiliki alergi terhadap kacang dan seafood.",
        "Proyek riset saya menggunakan model Phi-3 dan all-MiniLM-L6-v2.",
        "Saya lebih menyukai jawaban dalam bahasa Indonesia formal.",
    ]
    vecs = rng.standard_normal((len(interactions), 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    for i, text in enumerate(interactions):
        clock.current_turn = float(i * 5)
        index.add(vecs[i:i+1], texts=[text])

    index.fit(vecs)
    clock.current_turn = 50.0  # Current turn is 50

    # Initialize injector with 256 context token budget
    injector = LlamaCppMemoryInjector(
        index=index,
        max_context_tokens=256,
        chars_per_token=4.0,
    )

    # Current user message
    user_prompt = "Rekomendasikan menu makan malam untuk saya!"
    query_vec = vecs[1] + rng.normal(0, 0.02, 384).astype(np.float32)  # Relevant to allergies

    formatted_llm_prompt = injector.format_prompt(
        user_message=user_prompt,
        query_vec=query_vec,
        k=3,
        system_prefix="Anda adalah asisten AI lokal yang cerdas dan hemat memori.",
    )

    print("\n--- Injected Prompt for Llama.cpp / Ollama ---")
    print(formatted_llm_prompt)

if __name__ == "__main__":
    main()
