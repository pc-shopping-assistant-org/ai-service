from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_service.api.router import api_router
from ai_service.config.enums import StatefulReactVersion
from ai_service.config.settings import Settings, get_settings
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        ErrorDetail(
            code="VALIDATION_ERROR",
            message=str(error.get("msg", "Invalid value")),
            field=".".join(str(part) for part in error.get("loc", ()) if part != "body")
            or None,
        )
        for error in exc.errors()
    ]
    payload = ApiResponse[None](
        data=None, message=ResponseMessage.VALIDATION_ERROR, errors=errors
    )
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


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
    return JSONResponse(
        status_code=exc.status_code, content=payload.model_dump(mode="json")
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    payload = ApiResponse[None](
        data=None,
        message=ResponseMessage.INTERNAL_SERVER_ERROR,
        errors=[
            ErrorDetail(
                code=ResponseMessage.INTERNAL_SERVER_ERROR.value,
                message="Internal server error",
            )
        ],
    )
    return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.stateful = None
        if not runtime_settings.stateful_enabled:
            yield
            return
        from ai_service.infrastructure.stateful import open_stateful_resources

        async with open_stateful_resources(runtime_settings) as resources:
            application.state.stateful = resources
            try:
                yield
            finally:
                application.state.stateful = None

    application = FastAPI(
        title="PC Shopping Assistant AI Service",
        description="AI service for PC component recommendations and compatibility checks.",
        version="0.1.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "health",
                "description": "Service health and readiness endpoints.",
            },
            {
                "name": "assistant",
                "description": "Catalog-grounded assistant workflows, including SSE chat streaming.",
            },
        ],
    )
    application.include_router(api_router, prefix="/api/v1")
    application.exception_handler(RequestValidationError)(validation_exception_handler)
    application.exception_handler(StarletteHTTPException)(http_exception_handler)
    application.exception_handler(Exception)(unhandled_exception_handler)
    if runtime_settings.stateful_enabled:
        from ai_service.api.stateful_conversations import (
            StatefulApiError,
            create_stateful_router,
            stateful_exception_handler,
        )

        application.exception_handler(StatefulApiError)(stateful_exception_handler)
        application.include_router(
            create_stateful_router(
                2
                if runtime_settings.stateful_react_version == StatefulReactVersion.V2
                else 1
            ),
            prefix="/api/v1",
        )
    return application


app = create_app()


def run() -> None:
    uvicorn.run("ai_service.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
