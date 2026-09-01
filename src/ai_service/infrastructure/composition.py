"""Application composition root.

All runtime dependency wiring belongs here. FastAPI adapters receive the
application port instead of constructing providers, retrievers or graphs at
module scope. Tests can build the same container with a custom ``Settings``
and override individual dependencies directly on ``AssistantService``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_service.application.ports.answer_generator import AnswerGenerator
from ai_service.application.ports.assistant import AssistantUseCase
from ai_service.application.ports.catalog import CatalogRetriever
from ai_service.application.ports.graph_runner import GraphRunner
from ai_service.application.use_cases.assistant import AssistantService
from ai_service.capabilities.assistant.graphs.comparison import (
    ComparisonInput,
    ComparisonOutput,
    ComparisonState,
    comparison_graph,
)
from ai_service.capabilities.assistant.graphs.shopping import (
    ShoppingInput,
    ShoppingOutput,
    ShoppingState,
    shopping_graph,
)
from ai_service.config.settings import Settings, get_settings
from ai_service.context.manager import ConversationManager
from ai_service.infrastructure.graph.pydantic_runner import PydanticGraphRunner
from ai_service.infrastructure.providers.pydantic_ai_answer_generator import (
    PydanticAIAnswerGenerator,
)
from ai_service.services.backend_client import BackendClient
from ai_service.services.semantic_retriever import build_catalog_retriever


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """Request-independent services and adapters for one process."""

    settings: Settings
    backend_client: BackendClient
    conversation_manager: ConversationManager
    catalog_retriever: CatalogRetriever
    answer_generator: AnswerGenerator
    shopping_graph_runner: GraphRunner[ShoppingInput, ShoppingOutput]
    comparison_graph_runner: GraphRunner[ComparisonInput, ComparisonOutput]
    assistant: AssistantUseCase

    @property
    def assistant_service(self) -> AssistantUseCase:
        """Compatibility name while callers migrate to the port-oriented name."""

        return self.assistant


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    """Build the default application graph without performing external I/O."""

    runtime_settings = settings or get_settings()
    backend_client = BackendClient(runtime_settings)
    conversation_manager = ConversationManager(runtime_settings)
    catalog_retriever = build_catalog_retriever(backend_client, runtime_settings)
    answer_generator = PydanticAIAnswerGenerator(runtime_settings)
    shopping_graph_runner: GraphRunner[ShoppingInput, ShoppingOutput] = PydanticGraphRunner(
        shopping_graph,
        state_factory=ShoppingState,
    )
    comparison_graph_runner: GraphRunner[ComparisonInput, ComparisonOutput] = PydanticGraphRunner(
        comparison_graph,
        state_factory=ComparisonState,
    )
    assistant = AssistantService(
        backend_client=backend_client,
        context_manager=conversation_manager,
        answer_generator=answer_generator,
        retriever=catalog_retriever,
        settings=runtime_settings,
        shopping_graph_runner=shopping_graph_runner,
        comparison_graph_runner=comparison_graph_runner,
    )
    return ApplicationContainer(
        settings=runtime_settings,
        backend_client=backend_client,
        conversation_manager=conversation_manager,
        catalog_retriever=catalog_retriever,
        answer_generator=answer_generator,
        shopping_graph_runner=shopping_graph_runner,
        comparison_graph_runner=comparison_graph_runner,
        assistant=assistant,
    )


__all__ = ["ApplicationContainer", "build_container"]
