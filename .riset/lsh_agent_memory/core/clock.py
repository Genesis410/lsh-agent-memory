"""Pluggable clock interface for recency calculation."""

import time
from typing import Protocol


class Clock(Protocol):
    """Protocol for abstract time sources."""

    def now(self) -> float:
        """Return current timestamp or turn counter."""
        ...

    def step(self, amount: float = 1.0) -> None:
        """Advance time if supported by clock implementation."""
        ...


class TurnIndexClock:
    """Discrete dialogue turn counter for deterministic benchmarks."""

    def __init__(self, start_turn: float = 0.0) -> None:
        self.current_turn = float(start_turn)

    def now(self) -> float:
        return self.current_turn

    def step(self, amount: float = 1.0) -> None:
        self.current_turn += amount


class WallClock:
    """Real-world POSIX epoch clock in seconds."""

    def now(self) -> float:
        return time.time()

    def step(self, amount: float = 1.0) -> None:
        # Wall clock is driven by system time
        pass
