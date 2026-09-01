from collections.abc import AsyncIterator


class DeterministicAnswerGenerator:
    """Application-safe fallback used when no external model is wired."""

    async def generate(self, prompt: str, fallback: str) -> str:
        del prompt
        return fallback

    async def stream(self, prompt: str, fallback: str) -> AsyncIterator[str]:
        del prompt
        yield fallback
