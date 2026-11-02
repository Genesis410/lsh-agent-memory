"""Scorer combining semantic cosine similarity with exponential recency decay."""

from typing import Optional
import numpy as np
from lsh_agent_memory.core.clock import Clock, TurnIndexClock


class EpisodicScorer:
    """Calculates episodic relevance score: alpha * cosine + (1 - alpha) * 2^(-delta_t / h)."""

    def __init__(
        self,
        alpha: float = 0.8,
        half_life: float = 200.0,
        clock: Optional[Clock] = None,
    ) -> None:
        """Initialize episodic scorer.

        Args:
            alpha: Weight of semantic similarity in [0.0, 1.0]. 1.0 = pure semantic.
            half_life: Recency half-life (h). Relevance drops to 50% after h time units.
            clock: Time source instance. Defaults to TurnIndexClock().
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if half_life <= 0:
            raise ValueError(f"half_life must be positive, got {half_life}")

        self.alpha = float(alpha)
        self.half_life = float(half_life)
        self.clock = clock if clock is not None else TurnIndexClock()

    def score_candidates(
        self,
        q_vec: np.ndarray,
        candidate_vecs: np.ndarray,
        candidate_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Score candidate vectors against query vector.

        Args:
            q_vec: Normalized query embedding of shape (d,).
            candidate_vecs: Candidate embeddings of shape (K, d).
            candidate_timestamps: Timestamps of candidates of shape (K,).

        Returns:
            Scores array of shape (K,).
        """
        if len(candidate_vecs) == 0:
            return np.empty(0, dtype=np.float32)

        # Cosine similarity (assuming vectors are L2-normalized)
        cos_sim = candidate_vecs @ q_vec

        if self.alpha >= 1.0:
            return cos_sim.astype(np.float32)

        # Recency decay
        now = self.clock.now()
        delta_t = np.maximum(0.0, now - candidate_timestamps)
        # 2^(-delta_t / h) = exp(-delta_t * (ln 2 / h))
        decay = np.exp2(-delta_t / self.half_life)

        # ponytail: linear alpha weighting with exponential decay. Upgrade path: non-linear sigmoidal fusion if prompt length dynamic.
        final_scores = (self.alpha * cos_sim) + ((1.0 - self.alpha) * decay)
        return final_scores.astype(np.float32)
