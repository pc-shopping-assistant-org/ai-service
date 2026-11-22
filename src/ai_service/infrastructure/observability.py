"""Run-scoped telemetry; no prompts, graph state or tool payloads are serialized."""

import asyncio
import json
import logging
import sys
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import (
    Span,
    SpanKind,
    StatusCode,
    Tracer,
    get_current_span,
    get_tracer,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.config.settings import Settings

SERVICE_NAME = "ai-service"
EXPORT_TIMEOUT_SECONDS = 5
EXPORT_INTERVAL_MS = 1000
LOG_FIELDS = (
    "event",
    "step",
    "duration_ms",
    "outcome",
    "result_status",
    "error_type",
    "error_location",
)
_PROPAGATOR = TraceContextTextMapPropagator()
logger = logging.getLogger(__name__)
_run: ContextVar[dict[str, str] | None] = ContextVar("ai_run_telemetry", default=None)
_tracer: ContextVar[Tracer | None] = ContextVar("ai_tracer", default=None)


@dataclass(frozen=True, slots=True)
class _StepMetrics:
    step_counter: Counter
    step_duration_ms: Histogram

    @classmethod
    def from_meter(cls, meter: Meter) -> "_StepMetrics":
        return cls(
            step_counter=meter.create_counter(
                name="ai_step",
                description="Count of AI service execution steps by outcome",
            ),
            step_duration_ms=meter.create_histogram(
                name="ai_step_duration_ms",
                unit="ms",
                description="Duration of AI service execution steps in milliseconds",
            ),
        )


_metrics: ContextVar[_StepMetrics | None] = ContextVar("ai_step_metrics", default=None)


class StepOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunIdGenerator(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        run_id = (_run.get() or {}).get("trace_id")
        return int(run_id, 16) if run_id else super().generate_trace_id()


def _extract_error_location(tb: TracebackType | None) -> str | None:
    """Return a PII-safe 'file:line:func' location from a traceback without exception text or locals."""
    if tb is None:
        return None
    frames = traceback.extract_tb(tb)
    if not frames:
        return None
    chosen = next(
        (
            frame
            for frame in reversed(frames)
            if "/ai_service/" in frame.filename.replace("\\", "/")
            and frame.filename != __file__
            and frame.name != "trace_http_requests"
        ),
        frames[-1],
    )
    normalized = chosen.filename.replace("\\", "/")
    marker = "/ai_service/"
    short_path = (
        normalized.split(marker, 1)[1]
        if marker in normalized
        else Path(normalized).name
    )
    return f"{short_path}:{chosen.lineno}:{chosen.name}"


def trace_headers() -> dict[str, str]:
    """Build W3C traceparent/tracestate headers for outbound HTTP calls when a span is active."""
    span = get_current_span().get_span_context()
    if not span.is_valid:
        return {}
    headers: dict[str, str] = {}
    _PROPAGATOR.inject(headers)
    return headers


def extract_trace_context(headers: Mapping[str, str]) -> Context:
    """Extract incoming W3C trace context from HTTP headers."""
    return _PROPAGATOR.extract(carrier={k.lower(): v for k, v in headers.items()})


@contextmanager
def http_request_span(
    method: str,
    path: str,
    headers: Mapping[str, str],
) -> Iterator[Span]:
    """Open a server span linked to any incoming W3C traceparent without recording payloads."""
    tracer = _tracer.get() or get_tracer(__name__)
    parent_ctx = extract_trace_context(headers)
    with tracer.start_as_current_span(
        f"HTTP {method} {path}",
        context=parent_ctx,
        kind=SpanKind.SERVER,
        attributes={
            "http.request.method": method,
            "url.path": path,
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        yield span


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
            error_location = _extract_error_location(record.exc_info[2])
            if error_location:
                record.error_location = error_location
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
    error_location: str | None = None
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
            observation.error_location = _extract_error_location(error.__traceback__)
            raise
        finally:
            if observation.outcome == StepOutcome.FAILED:
                span.set_status(StatusCode.ERROR)
            span.set_attribute("ai.outcome", observation.outcome)
            duration_ms = round((perf_counter() - started) * 1000, 3)
            extra: dict[str, str | float] = {
                "event": "step.finished",
                "step": name,
                "outcome": observation.outcome,
                "duration_ms": duration_ms,
            }
            if observation.error_type:
                span.set_attribute("error.type", observation.error_type)
                extra["error_type"] = observation.error_type
            if observation.error_location:
                span.set_attribute("error.location", observation.error_location)
                extra["error_location"] = observation.error_location
            if observation.result_status:
                span.set_attribute("ai.result_status", observation.result_status)
                extra["result_status"] = observation.result_status
            active_metrics = _metrics.get()
            if active_metrics is not None:
                metric_attrs = {"ai.step": name, "ai.outcome": observation.outcome.value}
                active_metrics.step_counter.add(1, metric_attrs)
                active_metrics.step_duration_ms.record(duration_ms, metric_attrs)
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
) -> Callable[P, Coroutine[Any, Any, R]]:
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
    metric_readers: list[MetricReader] = []
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
        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=f"{endpoint}/v1/metrics", timeout=EXPORT_TIMEOUT_SECONDS
                ),
                export_interval_millis=EXPORT_INTERVAL_MS,
            )
        )
        otlp_handler = LoggingHandler(logger_provider=logs)
        handlers.append(otlp_handler)
    metrics_provider = MeterProvider(
        resource=resource, metric_readers=metric_readers, shutdown_on_exit=False
    )
    step_metrics = _StepMetrics.from_meter(metrics_provider.get_meter(__name__))
    application_logger = logging.getLogger("ai_service")
    previous = application_logger.level, application_logger.propagate
    application_logger.setLevel(settings.log_level)
    application_logger.propagate = False
    for handler in handlers:
        handler.addFilter(CorrelationFilter())
        application_logger.addHandler(handler)
    metrics_token = _metrics.set(step_metrics)
    try:
        with telemetry_scope(traces.get_tracer(__name__)):
            yield
    finally:
        _metrics.reset(metrics_token)
        for handler in handlers:
            application_logger.removeHandler(handler)
            handler.close()
        metrics_provider.shutdown()
        logs.shutdown()
        traces.shutdown()
        application_logger.setLevel(previous[0])
        application_logger.propagate = previous[1]
