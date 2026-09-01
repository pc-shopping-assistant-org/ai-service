from uuid import UUID

import pytest

from ai_service.application.use_cases import AssistantService
from ai_service.capabilities.assistant.graphs.comparison import (
    ComparisonInput,
    ComparisonState,
    comparison_graph,
)
from ai_service.capabilities.assistant.graphs.shopping import (
    ShoppingInput,
    ShoppingState,
    shopping_graph,
)
from ai_service.config.enums import AIProvider, EmbeddingProviderKind, RetrievalBackend
from ai_service.config.settings import Settings
from ai_service.infrastructure.composition import build_container
from ai_service.infrastructure.graph import PydanticGraphRunner


@pytest.mark.asyncio
async def test_pydantic_graph_runner_keeps_state_request_scoped() -> None:
    runner = PydanticGraphRunner(
        shopping_graph,
        state_factory=ShoppingState,
    )

    first = await runner.run(ShoppingInput(query="  laptop   gaming  "))
    second = await runner.run(ShoppingInput(query="monitor"))

    assert first.query == "laptop gaming"
    assert second.query == "monitor"


@pytest.mark.asyncio
async def test_pydantic_graph_runner_supports_multiple_capability_graphs() -> None:
    runner = PydanticGraphRunner(
        comparison_graph,
        state_factory=ComparisonState,
    )
    product_ids = [
        UUID("00000000-0000-0000-0000-000000000001"),
        UUID("00000000-0000-0000-0000-000000000002"),
    ]

    result = await runner.run(ComparisonInput(product_ids=product_ids))

    assert result.product_ids == product_ids


def test_composition_root_wires_ports_without_external_io() -> None:
    container = build_container(Settings())

    assert isinstance(container.assistant, AssistantService)
    assert container.assistant_service is container.assistant
    assert container.settings.provider is AIProvider.FALLBACK
    assert container.settings.retrieval_backend is RetrievalBackend.BACKEND
    assert container.settings.embedding_provider is EmbeddingProviderKind.HTTP
    assert container.shopping_graph_runner is not None
    assert container.comparison_graph_runner is not None
