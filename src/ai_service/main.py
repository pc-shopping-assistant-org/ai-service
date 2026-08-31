import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_service.api.router import api_router
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage

app = FastAPI(
    title="PC Shopping Assistant AI Service",
    description="AI service for PC component recommendations and compatibility checks.",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "health",
            "description": "Service health and readiness endpoints.",
        },
    ],
)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        ErrorDetail(
            code="VALIDATION_ERROR",
            message=str(error.get("msg", "Invalid value")),
            field=".".join(str(part) for part in error.get("loc", ()) if part != "body") or None,
        )
        for error in exc.errors()
    ]
    payload = ApiResponse[None](data=None, message=ResponseMessage.VALIDATION_ERROR, errors=errors)
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    key, default_detail = {
        404: (ResponseMessage.ENDPOINT_NOT_FOUND, "API endpoint not found"),
        405: (ResponseMessage.METHOD_NOT_ALLOWED, "HTTP method is not allowed"),
    }.get(exc.status_code, (ResponseMessage.HTTP_ERROR, "HTTP request failed"))
    detail = exc.detail if isinstance(exc.detail, str) else default_detail
    payload = ApiResponse[None](
        data=None,
        message=key,
        errors=[ErrorDetail(code=key.value, message=detail)],
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    payload = ApiResponse[None](
        data=None,
        message=ResponseMessage.INTERNAL_SERVER_ERROR,
        errors=[ErrorDetail(code=ResponseMessage.INTERNAL_SERVER_ERROR.value, message="Internal server error")],
    )
    return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))


def run() -> None:
    uvicorn.run("ai_service.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
