"""Llama.cpp prompt context injector for episodic memory."""

from typing import List, Optional
import numpy as np
from lsh_agent_memory.core.index import LSHMemoryIndex


class LlamaCppMemoryInjector:
    """Formats and injects retrieved episodic memories into Llama.cpp prompt within a token budget."""

    def __init__(
        self,
        index: LSHMemoryIndex,
        max_context_tokens: int = 512,
        chars_per_token: float = 4.0,
    ) -> None:
        """Initialize prompt injector.

        Args:
            index: LSH episodic memory index instance.
            max_context_tokens: Maximum token budget for injected memory block.
            chars_per_token: Heuristic average character count per token.
        """
        self.index = index
        self.max_context_tokens = max_context_tokens
        self.max_chars = int(max_context_tokens * chars_per_token)

    def retrieve_context(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        pool_n: Optional[int] = None,
    ) -> List[dict]:
        """Fetch top-k memories formatted for prompt injection."""
        return self.index.query(query_vec, k=k, pool_n=pool_n)

    def format_prompt(
        self,
        user_message: str,
        query_vec: np.ndarray,
        k: int = 5,
        system_prefix: str = "Kamu adalah asisten AI yang teliti dan memiliki memori jangka panjang.",
    ) -> str:
        """Inject relevant episodic memories into prompt under the token budget.

        Args:
            user_message: Current incoming user prompt.
            query_vec: Vector embedding of user_message.
            k: Maximum number of memories to consider.
            system_prefix: System prompt instruction.

        Returns:
            Constructed prompt with memory section.
        """
        memories = self.retrieve_context(query_vec, k=k)

        memory_lines = []
        current_len = 0

        for m in memories:
            text = m.get("text", "").strip()
            if not text:
                continue
            line = f"- [Turn {int(m.get('timestamp', 0))}] {text}"
            if current_len + len(line) + 1 > self.max_chars:
                break
            memory_lines.append(line)
            current_len += len(line) + 1

        if memory_lines:
            context_block = "\n".join(memory_lines)
            prompt = (
                f"{system_prefix}\n\n"
                f"### Memori Konteks Relevan (Episodic Context):\n"
                f"{context_block}\n\n"
                f"### Percakapan Saat Ini:\n"
                f"User: {user_message}\n"
                f"Assistant:"
            )
        else:
            prompt = (
                f"{system_prefix}\n\n"
                f"### Percakapan Saat Ini:\n"
                f"User: {user_message}\n"
                f"Assistant:"
            )

        return prompt
