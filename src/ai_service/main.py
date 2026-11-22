import logging
from contextlib import asynccontextmanager
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry.trace import StatusCode
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_service.api.router import api_router
from ai_service.config.settings import Settings, get_settings
from ai_service.infrastructure.observability import StepOutcome, http_request_span
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage

logger = logging.getLogger(__name__)
_SKIP_TRACE_PATHS = frozenset({"/health", "/api/v1/health"})


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
    if not getattr(request.state, "_unhandled_logged", False):
        logger.error(
            "http.unhandled_exception method=%s path=%s",
            request.method,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={
                "event": "http.unhandled_exception",
                "step": f"HTTP {request.method} {request.url.path}",
                "outcome": StepOutcome.FAILED,
            },
        )
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
        runtime_settings.require_stateful_configuration()
        from ai_service.infrastructure.observability import open_telemetry
        from ai_service.infrastructure.stateful import open_stateful_resources

        with open_telemetry(runtime_settings):
            if runtime_settings.embedding_preload_on_startup:
                from ai_service.services.bge_m3 import get_bge_m3_service

                await get_bge_m3_service(runtime_settings).warmup()
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
    @application.middleware("http")
    async def trace_http_requests(request: Request, call_next):
        if request.url.path in _SKIP_TRACE_PATHS:
            return await call_next(request)
        started = perf_counter()
        step_name = f"HTTP {request.method} {request.url.path}"
        with http_request_span(request.method, request.url.path, request.headers) as span:
            try:
                response = await call_next(request)
            except BaseException as exc:
                duration_ms = round((perf_counter() - started) * 1000, 3)
                span.set_attribute("http.response.status_code", 500)
                span.set_status(StatusCode.ERROR)
                request.state._unhandled_logged = True
                logger.error(
                    "http.unhandled_exception method=%s path=%s",
                    request.method,
                    request.url.path,
                    exc_info=(type(exc), exc, exc.__traceback__),
                    extra={
                        "event": "http.unhandled_exception",
                        "step": step_name,
                        "outcome": StepOutcome.FAILED,
                        "duration_ms": duration_ms,
                        "result_status": "500",
                    },
                )
                raise
            duration_ms = round((perf_counter() - started) * 1000, 3)
            span.set_attribute("http.response.status_code", response.status_code)
            outcome = (
                StepOutcome.FAILED
                if response.status_code >= 500
                else StepOutcome.SUCCEEDED
            )
            if outcome == StepOutcome.FAILED:
                span.set_status(StatusCode.ERROR)
            logger.log(
                logging.ERROR if outcome == StepOutcome.FAILED else logging.INFO,
                "http.request.finished %s status=%s duration_ms=%s",
                step_name,
                response.status_code,
                duration_ms,
                extra={
                    "event": "http.request.finished",
                    "step": step_name,
                    "outcome": outcome,
                    "duration_ms": duration_ms,
                    "result_status": str(response.status_code),
                },
            )
            return response

    application.include_router(api_router, prefix="/api/v1")
    application.exception_handler(RequestValidationError)(validation_exception_handler)
    application.exception_handler(StarletteHTTPException)(http_exception_handler)
    application.exception_handler(Exception)(unhandled_exception_handler)
    from ai_service.api.stateful_conversations import (
        StatefulApiError,
        stateful_exception_handler,
    )
    from ai_service.api.stateful_conversations import (
        router as stateful_router,
    )

    application.exception_handler(StatefulApiError)(stateful_exception_handler)
    application.include_router(stateful_router, prefix="/api/v1")
    from ai_service.api.user_memory import router as memory_router

    application.include_router(memory_router, prefix="/api/v1")
    return application


app = create_app()


def run() -> None:
    uvicorn.run("ai_service.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
