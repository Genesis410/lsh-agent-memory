"""LSH-Agent-Memory: High-Throughput Episodic Memory for Local CPU LLMs.

Lightweight, zero-external-dependency episodic memory leveraging Data-Dependent
Locality-Sensitive Hashing (ITQ), vectorized SIMD POPCNT Hamming search, and
disk-backed memory-mapped Float16 storage with exponential recency decay.
"""

from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend, HammingBackend
from lsh_agent_memory.core.clock import Clock, TurnIndexClock, WallClock
from lsh_agent_memory.core.store import MemmapStore
from lsh_agent_memory.core.scorer import EpisodicScorer
from lsh_agent_memory.core.index import LSHMemoryIndex

__version__ = "0.1.0"

__all__ = [
    "fit_itq",
    "encode",
    "BitwiseCountBackend",
    "HammingBackend",
    "Clock",
    "TurnIndexClock",
    "WallClock",
    "MemmapStore",
    "EpisodicScorer",
    "LSHMemoryIndex",
    "__version__",
]
