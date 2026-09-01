from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_service.api.sse import encode_chat_event
from ai_service.infrastructure.providers.pydantic_ai_answer_generator import (
    PydanticAIAnswerGenerator,
)
from ai_service.schemas.conversation import (
    ChatData,
    ChatRequest,
    ChatStreamEvent,
    ChatStreamEventType,
    CompareData,
    CompareRequest,
    ConsultData,
    ConsultRequest,
    EvaluateData,
    EvaluateRequest,
    SearchData,
    SearchRequest,
)
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage
from ai_service.services.assistant_service import AssistantService

router = APIRouter()
assistant_service = AssistantService(answer_generator=PydanticAIAnswerGenerator())


@router.post("/chat", response_model=ApiResponse[ChatData], tags=["assistant"])
async def chat(request: ChatRequest) -> ApiResponse[ChatData]:
    return await assistant_service.chat(request)


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    tags=["assistant"],
    responses={
        200: {
            "description": "Server-sent chat events. Every data frame is a canonical API envelope.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in assistant_service.stream_chat(request):
                yield encode_chat_event(event)
        except Exception:  # noqa: BLE001 - keep SSE failures in the API contract
            error = ApiResponse(
                data=ChatStreamEvent(event=ChatStreamEventType.ERROR),
                message=ResponseMessage.AI_CHAT_STREAM_FAILED,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_CHAT_STREAM_FAILED.value,
                        message="Chat stream failed",
                    )
                ],
            )
            yield encode_chat_event(error)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
