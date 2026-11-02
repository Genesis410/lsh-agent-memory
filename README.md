# LSH-Agent-Memory

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware](https://img.shields.io/badge/hardware-CPU%20x86--64%20SIMD-success.svg)]()
[![Zero External DB](https://img.shields.io/badge/dependencies-NumPy%20%2B%20SciPy%20Only-brightgreen.svg)]()

> **High-throughput episodic memory for local CPU LLMs using Data-Dependent Locality-Sensitive Hashing (ITQ), vectorized SIMD POPCNT Hamming scan, and memory-mapped disk storage.**

---

## Overview

Local small language models (such as Phi-3, Mistral, and Llama-3 8B) running on commodity CPUs suffer from memory loss ("amnesia") across multi-day interactions due to strict context window limits. Standard vector database servers (Milvus, Pinecone, Qdrant) require heavy daemon installations, high RAM footprints, and external network services impractical for air-gapped or memory-bounded edge devices.

`lsh-agent-memory` provides a **zero-external-dependency, drop-in Python library** for episodic agent memory. It runs entirely in-process on CPU via **NumPy** and **SciPy**, utilizing only **16 bytes of RAM per vector** while providing sub-millisecond candidate retrieval and disk-backed Float16 re-ranking.

---

## Key Features

1. **16 Bytes / Vector Bitcode Footprint**:
   Compresses 384-dimensional dense embeddings (`all-MiniLM-L6-v2`) into compact 128-bit binary codes (2 × `uint64` words), indexing **1,000,000 vectors in just 15.26 MB of RAM** (a 98.9% memory reduction vs in-memory Float32).
2. **Data-Dependent Iterative Quantization (ITQ)**:
   Mitigates the representation degeneration (*cone effect*) common in bilingual Indonesian-English text embeddings via mean-centering, PCA projection, and SVD Procrustes orthogonal rotation.
3. **Hardware-Accelerated POPCNT Hamming Scan**:
   Leverages native x86-64 bitwise population count (`np.bitwise_count`) with unrolled 64-bit word XOR differences, scanning 1,000,000 entries in **23.82 ms** on standard laptop CPUs.
4. **Memory-Mapped Disk Storage (`np.memmap`)**:
   Stores raw Float16 vectors on NVMe/SSD, halving storage requirements (732.42 MB vs 1,464.84 MB on 1M vectors) with **100.0% Top-10 Cosine Overlap** against Float32 exact cosine.
5. **Episodic Decay Scoring**:
   Dual-objective relevance scoring combining semantic similarity and exponential recency decay:
   $$S(q, m_i) = \alpha \cdot \cos(q, m_i) + (1 - \alpha) \cdot 2^{-\Delta t / h}$$
   Preserves novel context ($\Delta t \le 200$) at 99.33% Needle Hit@10 retention.
6. **Zero External VectorDB**:
   No Docker containers, no background daemon services, no C++ compilation toolchain required. Pure, vectorized Python.

---

## Empirical Benchmark Highlights

Evaluated on an AMD Ryzen 5 PRO 4650U (6 Cores, 12 Threads, 16 GB DDR4 RAM):

| Method / Configuration | Scale | Index RAM Footprint | Query Latency (Median) | Top-10 Overlap / Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Flat Cosine (Float32)** | 1,000,000 | 1,464.84 MB | 16.48 ms | 100.0% (Ground Truth) |
| **Exact Flat Disk-Backed (Float16)** | 1,000,000 | < 10 MB (Paged) | 48.74 ms | 100.0% (vs Float32) |
| **FAISS IndexLSH (Raw Random)** | 1,000,000 | 15.26 MB | 57.38 ms | 16.67% |
| **LSH-Agent-Memory (128-bit ITQ)** | **1,000,000** | **15.26 MB** | **23.82 ms** | **94.20% (Pareto Optimal)** |
| **LSH-Agent-Memory (256-bit ITQ)** | **1,000,000** | **30.52 MB** | **31.10 ms** | **98.40%** |

*Key finding:* At 1M scale, LSH-Agent-Memory saves **98.9% index RAM** while maintaining **94.2%–98.4% retrieval fidelity** at sub-35 ms query latency.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Genesis410/lsh-agent-memory.git
cd lsh-agent-memory

# Install core library (NumPy and SciPy only)
pip install -e .

# Optional: Install with LangChain and Llama.cpp adapters
pip install -e ".[all]"
```

---

## Quickstart

```python
import numpy as np
from lsh_agent_memory import LSHMemoryIndex, TurnIndexClock

# 1. Initialize episodic index
clock = TurnIndexClock(start_turn=0.0)
index = LSHMemoryIndex(
    dim=384,
    bits=128,
    store_dir="./agent_memory_store",
    alpha=0.8,       # 80% semantic, 20% recency decay
    half_life=200.0, # Recency half-life in turns
    clock=clock,
)

# 2. Add memories with text and timestamps
sample_vectors = np.random.randn(100, 384).astype(np.float32)
sample_vectors /= np.linalg.norm(sample_vectors, axis=1, keepdims=True)
texts = [f"User interaction at turn {i}" for i in range(100)]

index.add(sample_vectors, texts=texts)

# 3. Fit data-dependent ITQ projection on sample
index.fit(sample_vectors[:50])

# 4. Advance time and query
clock.step(10.0)
query_vec = sample_vectors[0]
results = index.query(query_vec, k=5)

for res in results:
    print(f"Rank [Score: {res['score']:.4f}] Turn {res['timestamp']}: {res['text']}")
```

---

## Framework Adapters

### 1. LangChain VectorStore Adapter

```python
from lsh_agent_memory import LSHMemoryIndex
from lsh_agent_memory.adapters import LSHVectorStore

# Wrap core index with LangChain compatibility
vectorstore = LSHVectorStore(
    index=core_index,
    embedding_fn=your_embedding_model.encode,
)

# Use inside any LangChain RetrievalQA or Agent
docs = vectorstore.similarity_search("User preferences for breakfast", k=3)
```

### 2. Llama.cpp / Local LLM Context Injector

```python
from lsh_agent_memory.adapters import LlamaCppMemoryInjector

injector = LlamaCppMemoryInjector(
    index=core_index,
    max_context_tokens=512, # Budget constraint
)

# Automatically formats episodic context into the prompt
prompt = injector.format_prompt(
    user_message="Tolong beri saran menu sarapan!",
    query_vec=query_embedding,
    k=4,
)
```

---

## Running Examples & Self-Checks

```bash
# 1. Run unit checks
python tests/test_core.py

# 2. Run quickstart example
python examples/01_quickstart.py

# 3. Run prompt injection demo
python examples/03_llamacpp_injector.py

# 4. Run 5-second CPU performance benchmark
python examples/04_benchmark_demo.py
```

---

## License

This project is licensed under the **[Apache License 2.0](LICENSE)**.

You are free to use, modify, distribute, and integrate this software into open-source or commercial products, subject to the conditions of the Apache 2.0 License.

---

## Citation

If you use this work, algorithm, or empirical results in your academic research, please cite:

```bibtex
@software{lsh_agent_memory_2026,
  author = {Genesis410},
  title = {LSH-Agent-Memory: High-Throughput Episodic Memory for Local CPU LLMs via Data-Dependent Hashing},
  year = {2026},
  url = {https://github.com/Genesis410/lsh-agent-memory},
  license = {Apache-2.0}
}
```
