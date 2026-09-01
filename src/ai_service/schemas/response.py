from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ResponseMessage(StrEnum):
    """Stable top-level keys that the frontend maps to localized messages."""

    SUCCESS = "SUCCESS"
    HEALTH_OK = "HEALTH_OK"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    HTTP_ERROR = "HTTP_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    AI_CHAT_COMPLETED = "AI_CHAT_COMPLETED"
    AI_CHAT_STREAM_STARTED = "AI_CHAT_STREAM_STARTED"
    AI_CHAT_STREAM_DELTA = "AI_CHAT_STREAM_DELTA"
    AI_CHAT_STREAM_COMPLETED = "AI_CHAT_STREAM_COMPLETED"
    AI_CHAT_STREAM_FAILED = "AI_CHAT_STREAM_FAILED"
    AI_SEARCH_COMPLETED = "AI_SEARCH_COMPLETED"
    AI_SEARCH_NO_RESULTS = "AI_SEARCH_NO_RESULTS"
    AI_CONSULT_COMPLETED = "AI_CONSULT_COMPLETED"
    AI_COMPARE_COMPLETED = "AI_COMPARE_COMPLETED"
    AI_COMPARE_PARTIAL = "AI_COMPARE_PARTIAL"
    AI_EVALUATION_COMPLETED = "AI_EVALUATION_COMPLETED"
    AI_BACKEND_UNAVAILABLE = "AI_BACKEND_UNAVAILABLE"
    AI_PRODUCT_NOT_FOUND = "AI_PRODUCT_NOT_FOUND"


class ApiResponse[T](BaseModel):
    data: T | None = None
    message: ResponseMessage = Field(
        default=ResponseMessage.SUCCESS,
        description="Static frontend mapping key; request-specific details belong in errors",
        examples=["SUCCESS"],
    )
    errors: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    data: None = None
    message: ResponseMessage = Field(
        default=ResponseMessage.INTERNAL_SERVER_ERROR,
        description="Static frontend mapping key; request-specific details belong in errors",
        examples=["VALIDATION_ERROR"],
    )
    errors: list[ErrorDetail] = Field(default_factory=list)
