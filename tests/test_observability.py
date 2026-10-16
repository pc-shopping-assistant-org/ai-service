import asyncio
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_call_attempts import claim

from ai_service.capabilities.assistant.public_contracts import RunViewV1
from ai_service.config.settings import Settings
from ai_service.infrastructure.observability import (
    JsonLogFormatter,
    RunIdGenerator,
    execution_trace,
    observed_async,
    open_telemetry,
    step,
    telemetry_scope,
)


def test_configured_otlp_exports_correlated_logs_and_traces_and_flushes_on_close():
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    requests = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                (self.path, self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    logger = logging.getLogger("ai_service")
    previous = list(logger.handlers), logger.level, logger.propagate
    execution = claim()
    try:
        settings = Settings(
            _env_file=None, otlp_endpoint=f"http://127.0.0.1:{server.server_port}"
        )
        assert settings.otlp_endpoint is not None
        with open_telemetry(settings), execution_trace(execution), step("agent"):
            try:
                raise ValueError("PRIVATE_EXCEPTION_BODY")
            except ValueError:
                logger.exception("provider.failed")
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    assert (logger.handlers, logger.level, logger.propagate) == previous
    traces = [
        ExportTraceServiceRequest.FromString(body)
        for path, body in requests
        if path == "/v1/traces"
    ]
    logs = [
        ExportLogsServiceRequest.FromString(body)
        for path, body in requests
        if path == "/v1/logs"
    ]
    spans = [
        s
        for req in traces
        for r in req.resource_spans
        for scope in r.scope_spans
        for s in scope.spans
    ]
    records = [
        r
        for req in logs
        for resource in req.resource_logs
        for scope in resource.scope_logs
        for r in scope.log_records
    ]
    assert {s.name for s in spans} == {"run.execute", "agent"}
    assert len(records) == 5
    assert all(r.trace_id.hex() == execution.run_id.hex and r.span_id for r in records)
    assert "PRIVATE_EXCEPTION_BODY" not in str(logs)
    failure = next(r for r in records if r.body.string_value == "provider.failed")
    assert any(
        a.key == "error_type" and a.value.string_value == "ValueError"
        for a in failure.attributes
    )


@pytest.fixture
def telemetry():
    from opentelemetry.sdk.trace import TracerProvider

    exporter = InMemorySpanExporter()
    provider = TracerProvider(id_generator=RunIdGenerator())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("ai_service")
    level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with telemetry_scope(provider.get_tracer("test")):
            yield exporter, output
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        provider.shutdown()


def test_run_view_trace_reference_is_stable_and_round_trips():
    run_id = uuid4()
    run = RunViewV1(
        run_id=run_id,
        conversation_id=uuid4(),
        request_id=uuid4(),
        status="PENDING",
        base_revision=0,
    )
    assert run.trace_id == run_id.hex
    assert RunViewV1.model_validate(run.model_dump()).trace_id == run.trace_id


@pytest.mark.asyncio
async def test_graph_steps_share_run_trace_and_parent_without_payloads(telemetry):
    exporter, output = telemetry
    execution = claim()

    async def node(state):
        with step("tool.search_web"):
            await asyncio.sleep(0)
        return {"done": True}

    with execution_trace(execution):
        result = await observed_async("agent", node)({"secret": "do-not-log"})
    assert result == {"done": True}
    spans = exporter.get_finished_spans()
    assert {s.context.trace_id for s in spans} == {execution.run_id.int}
    by_name = {s.name: s for s in spans}
    assert by_name["agent"].parent.span_id == by_name["run.execute"].context.span_id
    assert by_name["tool.search_web"].parent.span_id == by_name["agent"].context.span_id
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records and all(r["trace_id"] == execution.run_id.hex for r in records)
    assert all(r["run_id"] == str(execution.run_id) and r["span_id"] for r in records)
    finished = [r for r in records if r["event"] == "step.finished"]
    assert len(finished) == 3 and all(r["duration_ms"] >= 0 for r in finished)
    assert "do-not-log" not in output.getvalue()


@pytest.mark.asyncio
async def test_failure_and_cancellation_are_timed_without_exception_text(telemetry):
    exporter, output = telemetry
    with execution_trace(claim()):
        with pytest.raises(ValueError), step("agent"):
            raise ValueError("API_KEY_PRIVATE")
        with pytest.raises(asyncio.CancelledError), step("read_web_page"):
            raise asyncio.CancelledError()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["agent"].status.is_ok is False
    assert spans["agent"].attributes["error.type"] == "ValueError"
    assert spans["read_web_page"].attributes["ai.outcome"] == "CANCELLED"
    assert "API_KEY_PRIVATE" not in output.getvalue()
    assert all(not s.events for s in spans.values())  # No raw exception stack/body.


@pytest.mark.asyncio
async def test_parallel_runs_isolate_context_and_retry_retains_trace(telemetry):
    exporter, output = telemetry
    first, second = claim(), claim()

    async def execute(execution):
        with execution_trace(execution):
            await asyncio.sleep(0)
            with step("agent"):
                await asyncio.sleep(0)

    await asyncio.gather(execute(first), execute(second))
    retry = first.model_copy(update={"execution_id": uuid4(), "generation": 2})
    await execute(retry)
    roots = [s for s in exporter.get_finished_spans() if s.name == "run.execute"]
    assert [s.context.trace_id for s in roots].count(first.run_id.int) == 2
    assert all(s.parent is None for s in roots)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert all(r["trace_id"] == r["run_id"].replace("-", "") for r in records)


@pytest.mark.asyncio
async def test_real_graph_instruments_nodes_and_tool_dispatch(telemetry):
    from test_flexible_graph import ReadModel, graph
    from test_react_tools import Catalog, context

    exporter, _output = telemetry
    with execution_trace(claim()):
        result = await graph(ReadModel(), Catalog(), context()).ainvoke(
            {"message": "GPU"}
        )
    assert result["phase"] == "READY_TO_PUBLISH"
    names = {s.name for s in exporter.get_finished_spans()}
    assert {
        "begin_turn",
        "agent",
        "validate_action",
        "execute_tool",
        "tool.search_catalog",
    } <= names


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_managed_execution_includes_finalize_and_records_actual_outcome(
    telemetry, fail
):
    from test_managed_executor import Coordinator

    from ai_service.infrastructure.execution.supervisor import ManagedExecutor

    exporter, output = telemetry
    execution = claim()
    coordinator = Coordinator(execution)
    finished = asyncio.Event()
    original = coordinator.stopped

    async def stopped(*args):
        await original(*args)
        finished.set()

    coordinator.stopped = stopped

    async def execute(current, children):
        with step("agent"):
            if fail:
                raise ValueError("PRIVATE_MESSAGE")

    executor = ManagedExecutor(
        coordinator, execute, worker_id=execution.worker_instance_id
    )
    await executor.start()
    try:
        await asyncio.wait_for(finished.wait(), 1)
    finally:
        assert await executor.stop(grace_seconds=1)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {"run.execute", "run.finalize", "agent"} <= spans.keys()
    assert spans["run.execute"].attributes["ai.outcome"] == (
        "FAILED" if fail else "SUCCEEDED"
    )
    assert "PRIVATE_MESSAGE" not in output.getvalue()
