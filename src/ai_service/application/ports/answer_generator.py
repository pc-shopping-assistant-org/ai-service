from collections.abc import AsyncIterator
from typing import Protocol


class AnswerGenerator(Protocol):
    """Application port for a complete grounded answer."""

    async def generate(self, prompt: str, fallback: str) -> str:
        """Return a model answer or the supplied deterministic fallback."""


class StreamingAnswerGenerator(Protocol):
    """Optional application port for incremental answer chunks.

    Chunks are deltas, not cumulative snapshots.  Implementations must yield
    the deterministic fallback when no provider is configured so callers can
    keep one predictable stream contract in local development.
    """

    def stream(self, prompt: str, fallback: str) -> AsyncIterator[str]:
        """Yield answer deltas until the answer is complete."""
