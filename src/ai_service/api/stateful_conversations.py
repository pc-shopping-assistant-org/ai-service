"""Owner-scoped foundation/run transport; no execution owned by HTTP requests."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field, StrictInt

from ai_service.application.ports.stateful_conversation import (
    ConversationNotFound,
    InvalidConversationCursor,
    RunAdmissionError,
)
from ai_service.capabilities.assistant.public_contracts import (
    ConversationCreateV1,
    ConversationViewV1,
    MessageViewV1,
    PageV1,
    RunViewV1,
)
from ai_service.capabilities.assistant.react.public_views import ConversationViewV2
from ai_service.capabilities.assistant.react.turn_contracts import (
    TurnRequest,
    TurnRequestV2,
)
from ai_service.capabilities.assistant.stateful_contracts import TurnRequestV1
from ai_service.infrastructure.auth import AuthenticationError
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.schemas.response import (
    ApiResponse,
    ErrorDetail,
    ErrorResponse,
    ResponseMessage,
)

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    responses={
        401: {"model": ErrorResponse, "description": "Verified access token required"},
        503: {"model": ErrorResponse, "description": "Stateful resources unavailable"},
    },
)
bearer = HTTPBearer(auto_error=False)


class StatefulApiError(Exception):
    def __init__(self, status_code: int, message: ResponseMessage) -> None:
        self.status_code = status_code
        self.message = message


async def stateful_exception_handler(
    request: Request, exc: StatefulApiError
) -> JSONResponse:
    details = {
        400: "Invalid cursor for this owner or conversation",
        401: "Authentication is required",
        404: "Conversation not found",
        503: "Stateful service is unavailable",
        422: "Conversation ID must match the route",
        409: "Run admission conflicts with the current conversation or request",
    }
    payload = ApiResponse[None](
        data=None,
        message=exc.message,
        errors=[
            ErrorDetail(
                code=exc.message.value,
                message=details[exc.status_code],
                field={400: "cursor", 422: "conversation_id"}.get(exc.status_code),
            )
        ],
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(mode="json"),
        headers={"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None,
    )


async def resources(request: Request) -> StatefulResources:
    value = getattr(request.app.state, "stateful", None)
    if value is None:
        raise StatefulApiError(503, ResponseMessage.HTTP_ERROR)
    return cast(StatefulResources, value)


async def account_id(
    runtime: Annotated[StatefulResources, Depends(resources)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> UUID:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise StatefulApiError(401, ResponseMessage.AI_UNAUTHENTICATED)
    try:
        return await runtime.authenticator.authenticate(credentials.credentials)
    except AuthenticationError:
        raise StatefulApiError(401, ResponseMessage.AI_UNAUTHENTICATED) from None


@router.post("", status_code=201, response_model=ApiResponse[ConversationViewV1])
async def create_conversation(
    body: ConversationCreateV1,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> ApiResponse[ConversationViewV1]:
    return ApiResponse(
        data=await runtime.conversations.create(owner, body.title),
        message=ResponseMessage.AI_CONVERSATION_CREATED,
    )


@router.get(
    "",
    response_model=ApiResponse[PageV1[ConversationViewV1]],
    responses={400: {"model": ErrorResponse}},
)
async def list_conversations(
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=36)] = None,
) -> ApiResponse[PageV1[ConversationViewV1] | PageV1[ConversationViewV2]]:
    try:
        data = await runtime.conversations.list(owner, limit, cursor)
    except InvalidConversationCursor:
        raise StatefulApiError(400, ResponseMessage.VALIDATION_ERROR) from None
    return ApiResponse(data=data, message=ResponseMessage.AI_CONVERSATIONS_LISTED)


@router.get(
    "/{conversation_id}",
    response_model=ApiResponse[ConversationViewV1],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Missing or non-owned conversation",
        }
    },
)
async def get_conversation(
    conversation_id: UUID,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> ApiResponse[ConversationViewV1]:
    try:
        data = await runtime.conversations.get(owner, conversation_id)
    except ConversationNotFound:
        raise StatefulApiError(404, ResponseMessage.AI_CONVERSATION_NOT_FOUND) from None
    return ApiResponse(data=data, message=ResponseMessage.AI_CONVERSATION_RETRIEVED)


@router.get(
    "/{conversation_id}/messages",
    response_model=ApiResponse[PageV1[MessageViewV1]],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def list_messages(
    conversation_id: UUID,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=36)] = None,
) -> ApiResponse[PageV1[MessageViewV1]]:
    try:
        page = await runtime.conversations.messages(
            owner, conversation_id, limit, cursor
        )
    except ConversationNotFound:
        raise StatefulApiError(404, ResponseMessage.AI_CONVERSATION_NOT_FOUND) from None
    except InvalidConversationCursor:
        raise StatefulApiError(400, ResponseMessage.VALIDATION_ERROR) from None
    return ApiResponse(data=page, message=ResponseMessage.AI_MESSAGES_LISTED)


def admission_error(exc: RunAdmissionError) -> StatefulApiError:
    key = ResponseMessage(str(exc))
    return StatefulApiError(
        404 if key == ResponseMessage.AI_CONVERSATION_NOT_FOUND else 409, key
    )


@router.post(
    "/{conversation_id}/runs",
    status_code=202,
    response_model=ApiResponse[RunViewV1],
    responses={409: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def submit_run(
    conversation_id: UUID,
    body: TurnRequestV1,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> ApiResponse[RunViewV1]:
    return await _submit_run(conversation_id, body, owner, runtime)


async def _submit_run(
    conversation_id: UUID, body: TurnRequest, owner: UUID, runtime: StatefulResources
) -> ApiResponse[RunViewV1]:
    if body.conversation_id != conversation_id:
        raise StatefulApiError(422, ResponseMessage.VALIDATION_ERROR)
    if runtime.runs is None:
        raise StatefulApiError(503, ResponseMessage.HTTP_ERROR)
    try:
        run = await runtime.runs.submit(owner, body)
    except RunAdmissionError as exc:
        raise admission_error(exc) from None
    return ApiResponse(data=run, message=ResponseMessage.AI_RUN_ACCEPTED)


class TurnRequestV2Body(TurnRequestV2):
    # Internal constructors default to V2; wire bodies must explicitly select it.
    schema_version: StrictInt = Field(ge=2, le=2)


async def submit_run_v2(
    conversation_id: UUID,
    body: TurnRequestV2Body,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> ApiResponse[RunViewV1]:
    return await _submit_run(conversation_id, body, owner, runtime)


@router.get(
    "/{conversation_id}/runs/{run_id}",
    response_model=ApiResponse[RunViewV1],
    responses={404: {"model": ErrorResponse}},
)
async def get_run(
    conversation_id: UUID,
    run_id: UUID,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> ApiResponse[RunViewV1]:
    if runtime.runs is None:
        raise StatefulApiError(503, ResponseMessage.HTTP_ERROR)
    try:
        run = await runtime.runs.get(owner, conversation_id, run_id)
    except RunAdmissionError as exc:
        raise admission_error(exc) from None
    return ApiResponse(data=run, message=ResponseMessage.AI_RUN_RETRIEVED)


@router.post(
    "/{conversation_id}/runs/{run_id}/cancel",
    status_code=202,
    response_model=ApiResponse[RunViewV1],
    responses={404: {"model": ErrorResponse}},
)
async def cancel_run(
    conversation_id: UUID,
    run_id: UUID,
    owner: Annotated[UUID, Depends(account_id)],
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> ApiResponse[RunViewV1]:
    if runtime.runs is None:
        raise StatefulApiError(503, ResponseMessage.HTTP_ERROR)
    try:
        run = await runtime.runs.cancel(owner, conversation_id, run_id)
    except RunAdmissionError as exc:
        raise admission_error(exc) from None
    return ApiResponse(data=run, message=ResponseMessage.AI_RUN_CANCEL_REQUESTED)


def create_stateful_router(version: int = 1) -> APIRouter:
    if version == 1:
        return router
    if version != 2:
        raise ValueError("Unknown stateful HTTP version")
    selected = APIRouter()
    for route in router.routes:
        if not isinstance(route, APIRoute) or route.methods is None:
            raise TypeError("Stateful routes must use typed FastAPI endpoints")
        selected.add_api_route(
            route.path,
            submit_run_v2 if route.endpoint is submit_run else route.endpoint,
            methods=list(route.methods),
            status_code=route.status_code,
            response_model=ApiResponse[ConversationViewV2]
            if route.endpoint in (create_conversation, get_conversation)
            else ApiResponse[PageV1[ConversationViewV2]]
            if route.endpoint is list_conversations
            else route.response_model,
            responses=route.responses,
            tags=route.tags,
            name=route.name,
        )
    return selected
