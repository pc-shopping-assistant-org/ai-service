from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ApiResponse[T](BaseModel):
    data: T
    message: str
    errors: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    data: None = None
    message: str
    errors: list[ErrorDetail] = Field(default_factory=list)
