"""Public lifecycle projections, sharing the coordinator's SQL transaction."""

from typing import Any

from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.public_contracts import (
    RunViewV1,
    StatefulErrorCode,
    TurnResultV1,
    to_public_consultation,
)
from ai_service.capabilities.assistant.react.contracts import ToolRecord
from ai_service.capabilities.assistant.react.public_views import (
    TurnResultV2,
    public_tool_products,
    to_public_flexible,
)
from ai_service.capabilities.assistant.schemas import ChatStreamEventType
from ai_service.capabilities.assistant.stateful_contracts import RunStatus

from .events import append_run_event


def public_run(run: RowMapping, status: str | None = None) -> RunViewV1:
    return RunViewV1.model_validate(
        {
            "run_id": run["id"],
            "conversation_id": run["conversation_id"],
            "request_id": run["request_id"],
            "status": status or run["status"],
            "base_revision": run["base_revision"],
        }
    )


async def emit_start(conn: AsyncConnection, run: RowMapping) -> None:
    await append_run_event(
        conn,
        run["id"],
        "start",
        {
            "event": ChatStreamEventType.START,
            "run": public_run(run).model_dump(mode="json"),
        },
    )


async def emit_terminal(
    conn: AsyncConnection,
    run: RowMapping,
    status: str,
    revision: int,
    values: dict[str, Any] | None = None,
) -> None:
    if status == RunStatus.FAILED_RETRYABLE:
        return
    if values is None:
        code = (
            StatefulErrorCode.RUN_CANCELLED
            if status == RunStatus.CANCELLED
            else StatefulErrorCode.RUN_FAILED
        )
        await append_run_event(
            conn,
            run["id"],
            "terminal",
            {"event": ChatStreamEventType.ERROR, "code": code.value},
        )
        return
    snapshot = AcceptedConversationSnapshot.model_validate(
        values["accepted_snapshot"]
        if "accepted_snapshot" in values
        else {
            "consultation": values["consultation"],
            "public_build": values["public_build"],
            "public_clarification": values.get("public_clarification"),
            "public_task_clarification": values.get("public_task_clarification"),
            "pending_tasks": values.get("pending_tasks"),
            "requirement_state": values.get("requirement_state"),
        }
    )
    data: dict[str, Any] = {
        "run": public_run(run, status),
        "revision": revision,
        "state": to_public_consultation(snapshot.consultation),
        "answer": values["answer"],
        "build": snapshot.public_build,
    }
    result: TurnResultV1
    if run["request_payload"].get("schema_version") == 2:
        if snapshot.flexible is None or snapshot.requirement_state is None:
            raise ValueError("V2 completion event requires accepted flexible state")
        selected = snapshot.flexible.response or snapshot.flexible.answer
        result = TurnResultV2(
            **data,
            flexible=to_public_flexible(snapshot.flexible, snapshot.requirement_state),
            products=public_tool_products(
                [ToolRecord.model_validate(row) for row in values.get("ledger", [])],
                selected.evidence_refs if selected else [],
            ),
        )
    else:
        result = TurnResultV1(**data)
    await append_run_event(
        conn,
        run["id"],
        "answer",
        {"event": ChatStreamEventType.DELTA, "delta": result.answer},
    )
    await append_run_event(
        conn,
        run["id"],
        "terminal",
        {
            "event": ChatStreamEventType.COMPLETED,
            "result": result.model_dump(mode="json"),
        },
    )
