import asyncio
from uuid import UUID

import pytest

from ai_service.application.use_cases import AssistantService
from ai_service.capabilities.assistant.graphs.comparison import (
    ComparisonInput,
    normalize_comparison,
)
from ai_service.capabilities.assistant.graphs.shopping import (
    ShoppingInput,
    normalize_shopping,
)
from ai_service.capabilities.assistant.schemas import AssistantIntent
from ai_service.config.enums import AIProvider, EmbeddingProviderKind, RetrievalBackend
from ai_service.config.settings import Settings
from ai_service.infrastructure.composition import build_container
from ai_service.infrastructure.graph import LangGraphRunner


@pytest.mark.asyncio
async def test_langgraph_runner_keeps_state_request_scoped() -> None:
    runner = LangGraphRunner(
        normalize_shopping,
    )

    first = await runner.run(ShoppingInput(query="  laptop   gaming  "))
    second = await runner.run(ShoppingInput(query="monitor"))

    assert first.query == "laptop gaming"
    assert second.query == "monitor"


@pytest.mark.asyncio
async def test_langgraph_concurrent_invocations_do_not_share_state() -> None:
    runner = LangGraphRunner(normalize_shopping)
    inputs = [ShoppingInput(query=f"  product   {i} ") for i in range(20)]
    outputs = await asyncio.gather(*(runner.run(item) for item in inputs))
    assert [item.query for item in outputs] == [f"product {i}" for i in range(20)]


@pytest.mark.asyncio
async def test_langgraph_runner_supports_multiple_capability_graphs() -> None:
    runner = LangGraphRunner(
        normalize_comparison,
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


def test_assistant_intent_is_a_stable_enum() -> None:
    assert AssistantService.classify_intent("tư vấn laptop") is AssistantIntent.CONSULT
