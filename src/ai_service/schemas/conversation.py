"""Compatibility exports for the assistant capability schemas.

New feature code should keep request/response models beside its capability;
this module remains for existing integrations that import the old path.
"""

from ai_service.capabilities.assistant.schemas import (
    ChatData,
    ChatRequest,
    ChatStreamEvent,
    ChatStreamEventType,
    CompareData,
    CompareRequest,
    ConsultData,
    ConsultRequest,
    ConversationMessage,
    EvaluateData,
    EvaluateRequest,
    ProductCard,
    ProductComparison,
    SearchData,
    SearchRequest,
)

__all__ = [
    "ChatData",
    "ChatRequest",
    "ChatStreamEvent",
    "ChatStreamEventType",
    "CompareData",
    "CompareRequest",
    "ConsultData",
    "ConsultRequest",
    "ConversationMessage",
    "EvaluateData",
    "EvaluateRequest",
    "ProductCard",
    "ProductComparison",
    "SearchData",
    "SearchRequest",
]
