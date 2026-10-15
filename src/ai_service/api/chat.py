from typing import Annotated

from fastapi import APIRouter, Depends

from ai_service.api.dependencies import get_assistant_service
from ai_service.application.ports.assistant import AssistantUseCase
from ai_service.capabilities.assistant.schemas import (
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

router = APIRouter()


@router.post("/search", response_model=ApiResponse[SearchData], tags=["assistant"])
async def search(
    request: SearchRequest,
    service: Annotated[AssistantUseCase, Depends(get_assistant_service)],
) -> ApiResponse[SearchData]:
    return await service.search(request)


@router.post("/consult", response_model=ApiResponse[ConsultData], tags=["assistant"])
async def consult(
    request: ConsultRequest,
    service: Annotated[AssistantUseCase, Depends(get_assistant_service)],
) -> ApiResponse[ConsultData]:
    return await service.consult(request)


@router.post("/compare", response_model=ApiResponse[CompareData | None], tags=["assistant"])
async def compare(
    request: CompareRequest,
    service: Annotated[AssistantUseCase, Depends(get_assistant_service)],
) -> ApiResponse[CompareData | None]:
    return await service.compare(request)


@router.post("/evaluate", response_model=ApiResponse[EvaluateData | None], tags=["assistant"])
async def evaluate(
    request: EvaluateRequest,
    service: Annotated[AssistantUseCase, Depends(get_assistant_service)],
) -> ApiResponse[EvaluateData | None]:
    return await service.evaluate(request)
