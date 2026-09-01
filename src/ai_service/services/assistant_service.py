"""Compatibility factory for the assistant application use case.

New code should import ``AssistantService`` from
``ai_service.application.use_cases``. Keeping this module avoids a needless
breaking change for existing tests and integrations while the package moves
to vertical capability slices.
"""

from ai_service.application.deterministic_answer_generator import (
    DeterministicAnswerGenerator,
)
from ai_service.application.ports.answer_generator import AnswerGenerator
from ai_service.application.ports.catalog import CatalogClient, CatalogRetriever
from ai_service.application.ports.conversation import ConversationStore
from ai_service.application.ports.graph_runner import GraphRunner
from ai_service.application.use_cases.assistant import (
    AssistantService as _AssistantService,
)
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
from ai_service.services.backend_client import BackendClient
from ai_service.services.semantic_retriever import build_catalog_retriever


class AssistantService(_AssistantService):
    """Backward-compatible default wiring for scripts/tests.

    Production HTTP wiring uses ``application.use_cases.AssistantService``
    directly from the composition root. This adapter keeps the old convenient
    constructor while callers migrate to explicit dependency injection.
    """

    def __init__(
        self,
        backend_client: CatalogClient | None = None,
        context_manager: ConversationStore | None = None,
        answer_generator: AnswerGenerator | None = None,
        retriever: CatalogRetriever | None = None,
        settings: Settings | None = None,
        shopping_graph_runner: GraphRunner[ShoppingInput, ShoppingOutput] | None = None,
        comparison_graph_runner: GraphRunner[ComparisonInput, ComparisonOutput] | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        backend = backend_client or BackendClient(runtime_settings)
        context = context_manager or ConversationManager(runtime_settings)
        catalog_retriever = retriever or build_catalog_retriever(backend, runtime_settings)
        shopping_runner = shopping_graph_runner or PydanticGraphRunner(
            shopping_graph,
            state_factory=ShoppingState,
        )
        comparison_runner = comparison_graph_runner or PydanticGraphRunner(
            comparison_graph,
            state_factory=ComparisonState,
        )
        super().__init__(
            backend_client=backend,
            context_manager=context,
            answer_generator=answer_generator or DeterministicAnswerGenerator(),
            retriever=catalog_retriever,
            settings=runtime_settings,
            shopping_graph_runner=shopping_runner,
            comparison_graph_runner=comparison_runner,
        )

__all__ = ["AssistantService"]
