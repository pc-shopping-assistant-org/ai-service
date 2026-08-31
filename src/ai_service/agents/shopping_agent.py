from pydantic_ai import Agent

from ai_service.schemas.agent import ShoppingAnswer


def create_shopping_agent(
    model: str,
    *,
    system_prompt: str | None = None,
) -> Agent[None, ShoppingAnswer]:
    """Create the shopping agent with an explicitly selected model."""
    return Agent(
        model=model,
        output_type=ShoppingAnswer,
        system_prompt=system_prompt or "You are a grounded PC shopping assistant.",
    )
