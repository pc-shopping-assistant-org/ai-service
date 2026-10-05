"""Stable application boundaries implemented by infrastructure adapters."""

from ai_service.application.ports.answer_generator import (
    AnswerGenerator,
    StreamingAnswerGenerator,
)
from ai_service.application.ports.assistant import AssistantUseCase
from ai_service.application.ports.catalog import (
    BackendCatalogClient,
    CatalogClient,
    CatalogPage,
    CatalogPageClient,
    CatalogRetriever,
)
from ai_service.application.ports.commerce import (
    CatalogFilter,
    CommerceClient,
    PromotionReport,
    VoucherOption,
)
from ai_service.application.ports.conversation import ConversationStore
from ai_service.application.ports.graph_runner import GraphRunner
from ai_service.application.ports.hardware import (
    CompatibilityIssue,
    CompatibilityReport,
    ComponentCategory,
    ComponentSpec,
    HardwareRuleEngine,
    RecommendedBuild,
    WattageReport,
)
from ai_service.application.ports.use_case import UseCase
from ai_service.application.ports.web_search import (
    WebSearchCategory,
    WebSearchClient,
    WebSearchItem,
    WebSearchQuery,
    WebSearchResult,
)

__all__ = [
    "AnswerGenerator",
    "AssistantUseCase",
    "BackendCatalogClient",
    "CatalogClient",
    "CatalogFilter",
    "CatalogPage",
    "CatalogPageClient",
    "CatalogRetriever",
    "CommerceClient",
    "CompatibilityIssue",
    "CompatibilityReport",
    "ComponentCategory",
    "ComponentSpec",
    "ConversationStore",
    "GraphRunner",
    "HardwareRuleEngine",
    "PromotionReport",
    "RecommendedBuild",
    "StreamingAnswerGenerator",
    "UseCase",
    "VoucherOption",
    "WattageReport",
    "WebSearchCategory",
    "WebSearchClient",
    "WebSearchItem",
    "WebSearchQuery",
    "WebSearchResult",
]


