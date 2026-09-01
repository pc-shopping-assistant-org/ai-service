from typing import Protocol


class ModelProvider(Protocol):
    """Port for a configured LLM model without leaking its SDK to the app."""

    @property
    def name(self) -> str:
        """Stable provider identifier used in logs and diagnostics."""

    def create_model(self) -> object:
        """Create a provider model lazily, on the first generation request."""
