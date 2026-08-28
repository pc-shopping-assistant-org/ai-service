from pydantic_ai import Agent

from ai_service.schemas.agent import ShoppingAnswer


def create_shopping_agent(model: str) -> Agent[None, ShoppingAnswer]:
    """Create the shopping agent with an explicitly selected model."""
    return Agent(model=model, output_type=ShoppingAnswer)
