"""Inbound assistant capability contract used by HTTP adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ai_service.capabilities.assistant.schemas import (
    ChatData,
    ChatRequest,
    ChatStreamEvent,
    CompareData,
    CompareRequest,
    ConsultData,
    ConsultRequest,
    EvaluateData,
    EvaluateRequest,
    SearchData,
    SearchRequest,
)
from ai_service.schemas.response import ApiResponse


class AssistantUseCase(Protocol):
    """Application operations exposed by the assistant capability."""

    async def chat(self, request: ChatRequest) -> ApiResponse[ChatData]:
        """Generate one complete assistant response."""

    def stream_chat(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ApiResponse[ChatStreamEvent]]:
        """Generate assistant response events incrementally."""

    async def search(self, request: SearchRequest) -> ApiResponse[SearchData]:
        """Search visible catalog products."""

    async def consult(self, request: ConsultRequest) -> ApiResponse[ConsultData]:
        """Recommend products for a natural-language need."""

    async def compare(
        self,
        request: CompareRequest,
    ) -> ApiResponse[CompareData | None]:
        """Compare a bounded set of products."""

    async def evaluate(
        self,
        request: EvaluateRequest,
    ) -> ApiResponse[EvaluateData | None]:
        """Explain one product using canonical catalog context."""


__all__ = ["AssistantUseCase"]
