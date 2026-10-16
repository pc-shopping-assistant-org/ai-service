"""Run-scoped telemetry; no prompts, graph state or tool payloads are serialized."""

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import wraps
from time import perf_counter

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import Span, StatusCode, Tracer, get_current_span, get_tracer

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.config.settings import Settings

SERVICE_NAME = "ai-service"
EXPORT_TIMEOUT_SECONDS = 5
EXPORT_INTERVAL_MS = 1000
LOG_FIELDS = ("event", "step", "duration_ms", "outcome", "result_status", "error_type")
logger = logging.getLogger(__name__)
_run: ContextVar[dict[str, str] | None] = ContextVar("ai_run_telemetry", default=None)
_tracer: ContextVar[Tracer | None] = ContextVar("ai_tracer", default=None)


class StepOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunIdGenerator(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        run_id = (_run.get() or {}).get("trace_id")
        return int(run_id, 16) if run_id else super().generate_trace_id()


def correlation() -> dict[str, str]:
    fields = dict(_run.get() or {})
    span = get_current_span().get_span_context()
    if span.is_valid:
        fields.update(trace_id=f"{span.trace_id:032x}", span_id=f"{span.span_id:016x}")
    return fields


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in correlation().items():
            setattr(record, key, value)
        if record.exc_info:
            error_type = record.exc_info[0]
            if error_type:
                record.error_type = error_type.__name__
            record.exc_info = None
            record.exc_text = None
        record.stack_info = None
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "service": SERVICE_NAME,
                "logger": record.name,
                "message": record.getMessage(),
                **correlation(),
                **{
                    key: getattr(record, key)
                    for key in LOG_FIELDS
                    if hasattr(record, key)
                },
            },
            ensure_ascii=False,
        )


@dataclass
class StepObservation:
    span: Span
    outcome: StepOutcome = StepOutcome.SUCCEEDED
    error_type: str | None = None
    result_status: str | None = None


@contextmanager
def step(name: str, *, root: bool = False) -> Iterator[StepObservation]:
    tracer = _tracer.get() or get_tracer(__name__)
    with tracer.start_as_current_span(
        name,
        context=Context() if root else None,
        attributes={"ai.step": name, **(_run.get() or {})},
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        observation = StepObservation(span)
        started = perf_counter()
        logger.info(
            "step.started %s", name, extra={"event": "step.started", "step": name}
        )
        try:
            yield observation
        except asyncio.CancelledError:
            observation.outcome = StepOutcome.CANCELLED
            raise
        except BaseException as error:
            observation.outcome = StepOutcome.FAILED
            observation.error_type = type(error).__name__
            raise
        finally:
            if observation.outcome == StepOutcome.FAILED:
                span.set_status(StatusCode.ERROR)
            span.set_attribute("ai.outcome", observation.outcome)
            extra: dict[str, str | float] = {
                "event": "step.finished",
                "step": name,
                "outcome": observation.outcome,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
            if observation.error_type:
                span.set_attribute("error.type", observation.error_type)
                extra["error_type"] = observation.error_type
            if observation.result_status:
                span.set_attribute("ai.result_status", observation.result_status)
                extra["result_status"] = observation.result_status
            logger.log(
                logging.ERROR
                if observation.outcome == StepOutcome.FAILED
                else logging.INFO,
                "step.finished %s outcome=%s duration_ms=%s",
                name,
                observation.outcome,
                extra["duration_ms"],
                extra=extra,
            )


@contextmanager
def execution_trace(claim: ExecutionClaim) -> Iterator[StepObservation]:
    token = _run.set(
        {
            "trace_id": claim.run_id.hex,
            "run_id": str(claim.run_id),
            "conversation_id": str(claim.conversation_id),
            "execution_id": str(claim.execution_id),
            "generation": str(claim.generation),
        }
    )
    try:
        with step("run.execute", root=True) as observation:
            yield observation
    finally:
        _run.reset(token)


def observed_async[**P, R](
    name: str,
    function: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    @wraps(function)
    async def observed(*args: P.args, **kwargs: P.kwargs) -> R:
        with step(name):
            return await function(*args, **kwargs)

    return observed


def observed_sync[**P, R](name: str, function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def observed(*args: P.args, **kwargs: P.kwargs) -> R:
        with step(name):
            return function(*args, **kwargs)

    return observed


@contextmanager
def telemetry_scope(tracer: Tracer) -> Iterator[None]:
    token = _tracer.set(tracer)
    try:
        yield
    finally:
        _tracer.reset(token)


@contextmanager
def open_telemetry(settings: Settings) -> Iterator[None]:
    """Start before the executor; flush only after its shutdown/drain."""
    resource = Resource.create({"service.name": SERVICE_NAME})
    traces = TracerProvider(
        resource=resource, id_generator=RunIdGenerator(), shutdown_on_exit=False
    )
    logs = LoggerProvider(resource=resource, shutdown_on_exit=False)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonLogFormatter())
    handlers: list[logging.Handler] = [console]
    if settings.otlp_endpoint:
        endpoint = str(settings.otlp_endpoint).rstrip("/")
        traces.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=f"{endpoint}/v1/traces", timeout=EXPORT_TIMEOUT_SECONDS
                ),
                schedule_delay_millis=EXPORT_INTERVAL_MS,
            )
        )
        logs.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(
                    endpoint=f"{endpoint}/v1/logs", timeout=EXPORT_TIMEOUT_SECONDS
                ),
                schedule_delay_millis=EXPORT_INTERVAL_MS,
            )
        )
        otlp_handler = LoggingHandler(logger_provider=logs)
        handlers.append(otlp_handler)
    application_logger = logging.getLogger("ai_service")
    previous = application_logger.level, application_logger.propagate
    application_logger.setLevel(settings.log_level)
    application_logger.propagate = False
    for handler in handlers:
        handler.addFilter(CorrelationFilter())
        application_logger.addHandler(handler)
    try:
        with telemetry_scope(traces.get_tracer(__name__)):
            yield
    finally:
        for handler in handlers:
            application_logger.removeHandler(handler)
            handler.close()
        logs.shutdown()
        traces.shutdown()
        application_logger.setLevel(previous[0])
        application_logger.propagate = previous[1]
