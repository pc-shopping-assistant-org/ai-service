"""Explicit authenticated memory management; no model can silently write memory."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from ai_service.api.stateful_conversations import (
    StatefulApiError,
    account_id,
    resources,
)
from ai_service.application.ports.user_memory import (
    MemoryEntry,
    MemoryInput,
    MemoryKey,
    MemoryLimitReached,
    UserMemoryStore,
)
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.schemas.response import ApiResponse, ErrorResponse, ResponseMessage

router = APIRouter(
    prefix="/memories",
    tags=["memory"],
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


async def memory_store(
    runtime: Annotated[StatefulResources, Depends(resources)],
) -> UserMemoryStore:
    if runtime.memories is None:
        raise StatefulApiError(503, ResponseMessage.HTTP_ERROR)
    return runtime.memories


@router.get("", response_model=ApiResponse[list[MemoryEntry]])
async def list_memories(
    owner: Annotated[UUID, Depends(account_id)],
    store: Annotated[UserMemoryStore, Depends(memory_store)],
) -> ApiResponse[list[MemoryEntry]]:
    return ApiResponse(
        data=await store.list(owner), message=ResponseMessage.AI_MEMORIES_LISTED
    )


@router.put(
    "/{key}",
    response_model=ApiResponse[MemoryEntry],
    responses={409: {"model": ErrorResponse}},
)
async def save_memory(
    key: MemoryKey,
    body: MemoryInput,
    owner: Annotated[UUID, Depends(account_id)],
    store: Annotated[UserMemoryStore, Depends(memory_store)],
) -> ApiResponse[MemoryEntry]:
    try:
        entry = await store.put(owner, key, body)
    except MemoryLimitReached:
        raise StatefulApiError(409, ResponseMessage.AI_MEMORY_LIMIT_REACHED) from None
    return ApiResponse(data=entry, message=ResponseMessage.AI_MEMORY_SAVED)


@router.delete("/{key}", response_model=ApiResponse[None])
async def forget_memory(
    key: MemoryKey,
    owner: Annotated[UUID, Depends(account_id)],
    store: Annotated[UserMemoryStore, Depends(memory_store)],
) -> ApiResponse[None]:
    await store.delete(owner, key)
    return ApiResponse(data=None, message=ResponseMessage.AI_MEMORY_DELETED)
