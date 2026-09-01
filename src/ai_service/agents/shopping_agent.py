from pydantic_ai import Agent
from pydantic_ai.models import Model

from ai_service.schemas.agent import ShoppingAnswer


def create_shopping_agent(
    model: str | Model,
    *,
    system_prompt: str | None = None,
) -> Agent[None, ShoppingAnswer]:
    """Create the shopping agent with an explicitly selected model."""
    return Agent(
        model=model,
        output_type=ShoppingAnswer,
        system_prompt=system_prompt or "You are a grounded PC shopping assistant.",
    )


def create_streaming_shopping_agent(
    model: str | Model,
    *,
    system_prompt: str | None = None,
) -> Agent[None, str]:
    """Create a text-output agent suitable for token/delta streaming.

    The regular shopping agent intentionally returns ``ShoppingAnswer`` for
    validated non-streaming responses.  PydanticAI cannot use
    ``stream_text()`` with a structured output schema, so the streaming port
    uses a text-only agent and keeps the same grounding instructions.
    """

    return Agent(
        model=model,
        output_type=str,
        system_prompt=system_prompt or "You are a grounded PC shopping assistant.",
    )
