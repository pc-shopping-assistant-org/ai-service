from __future__ import annotations

from uuid import UUID

import pytest

from ai_service.config.settings import Settings
from ai_service.context.manager import ConversationManager
from ai_service.schemas.conversation import (
    ChatRequest,
    CompareRequest,
    ConsultRequest,
    SearchRequest,
)
from ai_service.services.answer_generator import PydanticAIAnswerGenerator
from ai_service.services.assistant_service import AssistantService
from ai_service.services.retriever import BackendCatalogRetriever


class FakeBackend:
    async def search_products(self, query: str, limit: int = 10) -> list[dict]:
        return [{"id": "00000000-0000-0000-0000-000000000001", "name": f"{query} PC"}][:limit]

    async def get_product(self, product_id: UUID) -> dict | None:
        return {"id": str(product_id), "name": "Test PC", "description": "A test product"}


class FakeAnswerGenerator:
    def __init__(self, answer: str = "model answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def generate(self, prompt: str, fallback: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeStreamingAnswerGenerator(FakeAnswerGenerator):
    async def stream(self, prompt: str, fallback: str):
        self.prompts.append(prompt)
        yield "streamed "
        yield "answer"


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int) -> list[dict]:
        self.calls.append((query, limit))
        return [{"id": "00000000-0000-0000-0000-000000000001", "name": "Retrieved PC"}]


@pytest.mark.asyncio
async def test_chat_uses_static_message_and_keeps_context() -> None:
    manager = ConversationManager()
    service = AssistantService(FakeBackend(), manager)

    response = await service.chat(ChatRequest(message="tìm laptop gaming"))

    assert response.message == "AI_CHAT_COMPLETED"
    assert response.errors == []
    assert response.data is not None
    assert response.data.intent == "SEARCH"
    assert len(manager.history(response.data.conversation_id)) == 2


@pytest.mark.asyncio
async def test_stream_chat_emits_enveloped_deltas_and_persists_complete_answer() -> None:
    generator = FakeStreamingAnswerGenerator()
    manager = ConversationManager()
    service = AssistantService(
        FakeBackend(),
        context_manager=manager,
        answer_generator=generator,
    )

    events = [event async for event in service.stream_chat(ChatRequest(message="tìm laptop"))]

    assert [event.data.event.value for event in events if event.data] == [
        "START",
        "DELTA",
        "DELTA",
        "COMPLETED",
    ]
    assert events[0].message == "AI_CHAT_STREAM_STARTED"
    assert events[1].data is not None
    assert events[1].data.delta == "streamed "
    assert events[-1].message == "AI_CHAT_STREAM_COMPLETED"
    assert events[-1].data is not None
    assert events[-1].data.result is not None
    assert events[-1].data.result.answer == "streamed answer"
    assert len(manager.history(events[-1].data.result.conversation_id)) == 2


@pytest.mark.asyncio
async def test_chat_planner_preserves_non_retrieval_intent() -> None:
    retriever = RecordingRetriever()
    response = await AssistantService(FakeBackend(), retriever=retriever).chat(
        ChatRequest(message="so sánh hai laptop này")
    )

    assert response.data is not None
    assert response.data.intent == "COMPARE"
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_configured_answer_generator_is_grounded_and_keeps_envelope_key() -> None:
    generator = FakeAnswerGenerator("grounded model answer")
    response = await AssistantService(FakeBackend(), answer_generator=generator).consult(
        ConsultRequest(query="laptop cho sinh vien")
    )

    assert response.message == "AI_CONSULT_COMPLETED"
    assert response.data is not None
    assert response.data.answer == "grounded model answer"
    assert generator.prompts
    assert "laptop cho sinh vien PC" in generator.prompts[0]


@pytest.mark.asyncio
async def test_pydantic_ai_generator_uses_deterministic_fallback_without_model() -> None:
    generator = PydanticAIAnswerGenerator(Settings(model_name=None))

    assert await generator.generate("ignored", "local fallback") == "local fallback"


@pytest.mark.asyncio
async def test_pydantic_ai_generator_can_run_with_builtin_test_model() -> None:
    generator = PydanticAIAnswerGenerator(Settings(model_name="test"))

    assert await generator.generate("say hello", "local fallback") == "a"


@pytest.mark.asyncio
async def test_pydantic_ai_generator_streams_with_builtin_test_model() -> None:
    generator = PydanticAIAnswerGenerator(Settings(model_name="test"))

    chunks = [chunk async for chunk in generator.stream("say hello", "local fallback")]

    assert "".join(chunks)
    assert chunks != ["local fallback"]


def test_model_provider_adapters_are_lazy_but_build_supported_models() -> None:
    from ai_service.infrastructure.providers.factory import build_model_provider

    openai_provider = build_model_provider(
        Settings(
            provider="openai",
            model_name="gpt-4o-mini",
            openai_api_key="test-key",
        )
    )
    gemini_provider = build_model_provider(
        Settings(
            provider="gemini",
            model_name="gemini-2.5-flash",
            gemini_api_key="test-key",
        )
    )

    assert openai_provider is not None
    assert openai_provider.name == "openai"
    assert openai_provider.create_model().model_name == "gpt-4o-mini"  # type: ignore[attr-defined]
    assert gemini_provider is not None
    assert gemini_provider.name == "gemini"
    assert gemini_provider.create_model().model_name == "gemini-2.5-flash"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_search_returns_static_no_result_key_when_catalog_is_empty() -> None:
    class EmptyBackend(FakeBackend):
        async def search_products(self, query: str, limit: int = 10) -> list[dict]:
            return []

    response = await AssistantService(EmptyBackend()).search(SearchRequest(query="unknown"))

    assert response.message == "AI_SEARCH_NO_RESULTS"
    assert response.data is not None
    assert response.data.products == []


@pytest.mark.asyncio
async def test_search_uses_injected_retriever_boundary() -> None:
    retriever = RecordingRetriever()

    response = await AssistantService(
        FakeBackend(),
        retriever=retriever,
    ).search(SearchRequest(query="laptop cho sinh vien", limit=7))

    assert response.message == "AI_SEARCH_COMPLETED"
    assert response.data is not None
    assert [product.name for product in response.data.products] == ["Retrieved PC"]
    assert retriever.calls == [("laptop cho sinh vien", 7)]


@pytest.mark.asyncio
async def test_shopping_graph_normalizes_query_before_retrieval() -> None:
    retriever = RecordingRetriever()

    response = await AssistantService(
        FakeBackend(),
        retriever=retriever,
    ).consult(ConsultRequest(query="  laptop   cho   sinh vien  "))

    assert response.data is not None
    assert response.data.query == "laptop cho sinh vien"
    assert retriever.calls == [("laptop cho sinh vien", 5)]


@pytest.mark.asyncio
async def test_backend_catalog_retriever_delegates_to_canonical_client() -> None:
    backend = FakeBackend()

    products = await BackendCatalogRetriever(backend).search("gaming", 3)

    assert products[0]["name"] == "gaming PC"


@pytest.mark.asyncio
async def test_backend_catalog_retriever_expands_meaningful_terms_when_phrase_misses() -> None:
    class TokenBackend:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def search_products(self, query: str, limit: int = 10) -> list[dict]:
            self.queries.append(query)
            if query == "laptop cho sinh viên":
                return []
            return [{"id": query, "name": f"{query} PC"}]

    backend = TokenBackend()
    products = await BackendCatalogRetriever(backend).search("laptop cho sinh viên", 2)

    assert backend.queries == ["laptop cho sinh viên", "laptop", "sinh", "viên"]
    assert [product["name"] for product in products] == ["laptop PC", "sinh PC"]


@pytest.mark.asyncio
async def test_compare_requires_and_returns_requested_products() -> None:
    first = UUID("00000000-0000-0000-0000-000000000001")
    second = UUID("00000000-0000-0000-0000-000000000002")
    response = await AssistantService(FakeBackend()).compare(
        CompareRequest(product_ids=[first, second])
    )

    assert response.message == "AI_COMPARE_COMPLETED"
    assert response.data is not None
    assert len(response.data.products) == 2


def test_product_card_uses_lowest_variant_list_price_for_detail_payload() -> None:
    card = AssistantService._card(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Detail PC",
            "variants": [
                {"listPrice": 2000000},
                {"listPrice": 1500000},
            ],
        }
    )

    assert card.list_price == 1500000


def test_chat_search_does_not_misclassify_product_name_as_evaluation() -> None:
    assert AssistantService.classify_intent("tìm Review Product") == "SEARCH"
