from fastapi import APIRouter

from ai_service.schemas.conversation import (
    ChatData,
    ChatRequest,
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
from ai_service.services.assistant_service import AssistantService

router = APIRouter()
assistant_service = AssistantService()


@router.post("/chat", response_model=ApiResponse[ChatData], tags=["assistant"])
async def chat(request: ChatRequest) -> ApiResponse[ChatData]:
    return await assistant_service.chat(request)


@router.post("/search", response_model=ApiResponse[SearchData], tags=["assistant"])
async def search(request: SearchRequest) -> ApiResponse[SearchData]:
    return await assistant_service.search(request)


@router.post("/consult", response_model=ApiResponse[ConsultData], tags=["assistant"])
async def consult(request: ConsultRequest) -> ApiResponse[ConsultData]:
    return await assistant_service.consult(request)


@router.post("/compare", response_model=ApiResponse[CompareData | None], tags=["assistant"])
async def compare(request: CompareRequest) -> ApiResponse[CompareData | None]:
    return await assistant_service.compare(request)


@router.post("/evaluate", response_model=ApiResponse[EvaluateData | None], tags=["assistant"])
async def evaluate(request: EvaluateRequest) -> ApiResponse[EvaluateData | None]:
    return await assistant_service.evaluate(request)
