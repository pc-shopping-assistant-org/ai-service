from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from ai_service.schemas.response import ApiResponse, ResponseMessage
from ai_service.services.bge_m3 import BgeM3EmbeddingService, get_bge_m3_service

router = APIRouter()

MAX_EMBEDDING_BATCH_SIZE = 64
MAX_EMBEDDING_TEXT_LENGTH = 8000


class EmbeddingRequest(BaseModel):
    model: str = Field(default="BAAI/bge-m3", min_length=1, max_length=120)
    input: str | list[str]

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: str | list[str]) -> list[str]:
        items = [value] if isinstance(value, str) else list(value)
        if not items:
            raise ValueError("input must not be empty")
        if len(items) > MAX_EMBEDDING_BATCH_SIZE:
            raise ValueError(f"input batch size must not exceed {MAX_EMBEDDING_BATCH_SIZE}")
        cleaned: list[str] = []
        for idx, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"input[{idx}] must be a non-empty string")
            trimmed = item.strip()
            if len(trimmed) > MAX_EMBEDDING_TEXT_LENGTH:
                trimmed = trimmed[:MAX_EMBEDDING_TEXT_LENGTH]
            cleaned.append(trimmed)
        return cleaned


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponseData(BaseModel):
    model: str
    data: list[EmbeddingItem]
    embedding: list[float]
    device: str | None = None
    latency_ms: float | None = None


def get_embedding_service() -> BgeM3EmbeddingService:
    return get_bge_m3_service()


@router.post("/embeddings", response_model=ApiResponse[EmbeddingResponseData])
async def create_embeddings(
    payload: EmbeddingRequest,
    service: Annotated[BgeM3EmbeddingService, Depends(get_embedding_service)],
) -> ApiResponse[EmbeddingResponseData]:
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    vectors = await service.embed_texts(texts)
    runtime_status = service.status()
    items = [
        EmbeddingItem(index=index, embedding=vector)
        for index, vector in enumerate(vectors)
    ]
    return ApiResponse(
        data=EmbeddingResponseData(
            model=runtime_status.model_name,
            data=items,
            embedding=vectors[0] if vectors else [],
            device=runtime_status.device,
            latency_ms=runtime_status.last_latency_ms,
        ),
        message=ResponseMessage.SUCCESS,
    )
