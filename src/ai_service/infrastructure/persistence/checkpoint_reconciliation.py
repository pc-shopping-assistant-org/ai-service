"""Stopped execution discovery, never latest-thread substitution or state copying.

Caller holds the conversation/run locks and has verified stored execution-stop
authority. A matching metadata filter alone is not proof: every candidate must
have validated ancestry, and the scoped history must have one unique tip.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequest
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1

from .checkpoint_lineage import require_checkpoint_lineage


async def reconcile_stopped_checkpoint(
    saver: BaseCheckpointSaver,
    claim: ExecutionClaim,
    base: CheckpointRefV1,
    request: TurnRequest,
    graph_version: str,
    *,
    stored: CheckpointRefV1 | None,
    recovery_sources: list[dict[str, Any]],
    limit: int = 256,
) -> CheckpointRefV1:
    if limit < 1:
        raise ValueError("Invalid checkpoint reconciliation bound")
    expected = {
        "app_run_id": str(claim.run_id),
        "app_execution_id": str(claim.execution_id),
        "app_generation": claim.generation,
        "app_base_checkpoint_id": base.checkpoint_id,
        "app_input_hash": canonical_hash(request.execution_input()),
        "app_graph_version": graph_version,
    }
    records = [
        record
        async for record in saver.alist(
            {
                "configurable": {
                    "thread_id": str(claim.conversation_id),
                    "checkpoint_ns": "",
                }
            },
            filter=expected,
            limit=limit + 1,
        )
    ]
    if not records or len(records) > limit:
        raise ValueError("Recovery checkpoint history is missing or exceeds bound")
    identifiers, parents = set(), set()
    for record in records:
        native = record.config["configurable"]
        if (
            native["thread_id"] != str(claim.conversation_id)
            or native.get("checkpoint_ns", "") != ""
        ):
            raise ValueError("Recovery candidate thread mismatch")
        identifiers.add(native["checkpoint_id"])
        if record.parent_config is not None:
            parents.add(record.parent_config["configurable"]["checkpoint_id"])
        await require_checkpoint_lineage(
            saver,
            record,
            claim,
            base,
            request,
            graph_version,
            recovery_sources=recovery_sources,
        )
    if stored is not None and (
        stored.thread_id != claim.conversation_id
        or stored.checkpoint_ns != ""
        or stored.checkpoint_id not in identifiers
    ):
        raise ValueError("Stored recovery cursor is outside this execution")
    tips = identifiers - parents
    if len(tips) != 1:
        raise ValueError("Recovery checkpoint lineage is ambiguous")
    return CheckpointRefV1(
        thread_id=claim.conversation_id, checkpoint_id=next(iter(tips))
    )
