"""Tests for ephemeral live agent activity status streaming."""

import asyncio
import json
from uuid import uuid4

import pytest

from ai_service.api.sse import run_event_frames
from ai_service.application.activity import RunActivityBroker
from ai_service.application.ports.run_events import StoredRunEvent
from ai_service.application.use_cases.run_events import StatefulRunEventService
from ai_service.capabilities.assistant.public_contracts import (
    AgentActivityStage,
    PublicConsultationV1,
    RunCompletedV1,
    RunStartedV1,
    RunViewV1,
    TurnResultV1,
)
from ai_service.capabilities.assistant.react.contracts import ToolCall, ToolName
from ai_service.capabilities.assistant.stateful_contracts import RunStatus
from ai_service.infrastructure.graph.tool_batches import tool_activity_meta
from ai_service.schemas.response import ApiResponse, ResponseMessage


def test_activity_broker_replays_history_dedups_and_cleans_up():
    broker = RunActivityBroker(max_per_run=4, max_runs=2)
    cid, rid = uuid4(), uuid4()
    emit = broker.sink(cid, rid)
    emit(AgentActivityStage.THINKING)
    emit(AgentActivityStage.THINKING)  # consecutive duplicate ignored
    emit(AgentActivityStage.SEARCHING_CATALOG, "search_catalog", "RTX 4070")

    with broker.subscribe(rid) as queue:
        first = queue.get_nowait()
        second = queue.get_nowait()
        assert queue.empty()
        assert first.stage == AgentActivityStage.THINKING
        assert second.stage == AgentActivityStage.SEARCHING_CATALOG
        assert second.tool == "search_catalog"
        assert second.detail == "RTX 4070"

    broker.finish(rid)
    with broker.subscribe(rid) as queue:
        assert queue.empty()


def test_tool_activity_meta_maps_stage_without_leaking_uuids():
    stage, detail = tool_activity_meta(
        ToolCall(
            call_id="c1",
            name=ToolName.SEARCH_CATALOG,
            arguments={"query": "RTX 4070 Super"},
        )
    )
    assert stage == AgentActivityStage.SEARCHING_CATALOG
    assert detail == "RTX 4070 Super"

    stage_details, detail_details = tool_activity_meta(
        ToolCall(
            call_id="c2",
            name=ToolName.GET_PRODUCT_DETAILS,
            arguments={"product_id": str(uuid4())},
        )
    )
    assert stage_details == AgentActivityStage.LOADING_DETAILS
    assert detail_details is None

    stage_cmp, detail_cmp = tool_activity_meta(
        ToolCall(
            call_id="c3",
            name=ToolName.COMPARE_PRODUCTS,
            arguments={"queries": ["RTX 4070", "RX 7800 XT"]},
        )
    )
    assert stage_cmp == AgentActivityStage.COMPARING
    assert detail_cmp == "RTX 4070 vs RX 7800 XT"


@pytest.mark.asyncio
async def test_sse_stream_interleaves_ephemeral_status_without_affecting_sequences():
    owner, cid, rid, req_id = (uuid4() for _ in range(4))
    broker = RunActivityBroker()
    start_view = RunViewV1(
        run_id=rid,
        conversation_id=cid,
        request_id=req_id,
        status=RunStatus.RUNNING,
        base_revision=0,
    )
    done_view = start_view.model_copy(update={"status": RunStatus.COMPLETED})
    start_row = StoredRunEvent(
        event_id=uuid4(),
        sequence=1,
        envelope=ApiResponse(
            data=RunStartedV1(
                conversation_id=cid, run_id=rid, sequence=1, event="START", run=start_view
            ),
            message=ResponseMessage.AI_CHAT_STREAM_STARTED,
        ),
    )
    done_row = StoredRunEvent(
        event_id=uuid4(),
        sequence=2,
        envelope=ApiResponse(
            data=RunCompletedV1(
                conversation_id=cid,
                run_id=rid,
                sequence=2,
                event="COMPLETED",
                result=TurnResultV1(
                    run=done_view,
                    revision=1,
                    state=PublicConsultationV1(
                        budget_scope="BUILD_PC",
                        target_budget_vnd=None,
                        accessory_budget_vnd=None,
                        use_case=None,
                        target_resolution=None,
                        target_fps=None,
                        preferred_gpu_brand=None,
                        preferred_cpu_brand=None,
                        form_factor_preference=None,
                        owned_parts={},
                        pinned_parts={},
                        accessories={},
                        build_stale=False,
                    ),
                    answer="Done",
                ),
            ),
            message=ResponseMessage.AI_CHAT_STREAM_COMPLETED,
        ),
    )
    journal_rows = [start_row]
    current_status = RunStatus.RUNNING

    class Reader:
        async def read(self, acct, conv, run, after=None, limit=100):
            if after is None:
                return list(journal_rows)
            idx = next(
                (i for i, row in enumerate(journal_rows) if row.event_id == after), -1
            )
            return journal_rows[idx + 1 :]

    class Runs:
        async def get(self, acct, conv, run):
            return start_view.model_copy(update={"status": current_status})

    service = StatefulRunEventService(
        Reader(),  # type: ignore[arg-type]
        Runs(),  # type: ignore[arg-type]
        activity_broker=broker,
        poll_seconds=0.02,
    )
    sub = await service.open(owner, cid, rid)

    async def produce():
        await asyncio.sleep(0.01)
        broker.publish(cid, rid, AgentActivityStage.THINKING)
        await asyncio.sleep(0.01)
        broker.publish(
            cid, rid, AgentActivityStage.SEARCHING_CATALOG, "search_catalog", "RTX 4070"
        )
        await asyncio.sleep(0.03)
        nonlocal current_status
        current_status = RunStatus.COMPLETED
        journal_rows.append(done_row)
        broker.finish(rid)

    producer = asyncio.create_task(produce())
    frames = [frame async for frame in run_event_frames(sub)]
    await producer

    parsed = []
    for frame in frames:
        if frame.startswith(":"):
            continue
        lines = frame.strip().splitlines()
        event_name = lines[0].split(": ", 1)[1]
        payload = json.loads(lines[1].split(": ", 1)[1])
        parsed.append((event_name, payload))

    names = [name for name, _ in parsed]
    assert names[0] == "start"
    assert "status" in names
    assert names[-1] == "completed"
    status_payloads = [p["data"] for n, p in parsed if n == "status"]
    assert [s["stage"] for s in status_payloads] == ["THINKING", "SEARCHING_CATALOG"]
    assert all("sequence" not in s for s in status_payloads)
    assert parsed[0][1]["data"]["sequence"] == 1
    assert parsed[-1][1]["data"]["sequence"] == 2
