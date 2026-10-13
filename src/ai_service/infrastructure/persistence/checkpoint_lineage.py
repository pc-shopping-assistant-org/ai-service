"""Native ancestry validation, including exact authorized recovery boundaries."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequest
from ai_service.capabilities.assistant.recovery_policy import (
    ExecutionIdentityV1,
    require_stopped_execution,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ExecutionStopEvidenceV1,
)


async def require_checkpoint_lineage(
    saver: BaseCheckpointSaver,
    record: CheckpointTuple,
    claim: ExecutionClaim,
    base: CheckpointRefV1,
    request: TurnRequest,
    graph_version: str | None = None,
    *,
    recovery_sources: list[dict[str, Any]] | None = None,
) -> None:
    expected = {
        "app_run_id": str(claim.run_id),
        "app_execution_id": str(claim.execution_id),
        "app_generation": claim.generation,
        "app_base_checkpoint_id": base.checkpoint_id,
        "app_input_hash": canonical_hash(request.execution_input()),
    }
    if graph_version:
        expected["app_graph_version"] = graph_version
    seen = set()
    authorized = {}
    for item in recovery_sources or []:
        ref = CheckpointRefV1.model_validate(item["source_ref"])
        previous = ExecutionClaim.model_validate(item["execution"])
        proof = ExecutionStopEvidenceV1.model_validate(item["stop_evidence"])
        require_stopped_execution(
            ExecutionIdentityV1(
                execution_id=previous.execution_id,
                worker_instance_id=previous.worker_instance_id,
                generation=previous.generation,
            ),
            proof,
        )
        if (
            ref.thread_id != claim.conversation_id
            or ref.checkpoint_ns != ""
            or previous.run_id != claim.run_id
            or previous.conversation_id != claim.conversation_id
            or item["graph_version"] != graph_version
            or proof.kind != "EXECUTOR_STOP_ACK"
            or ref.checkpoint_id in authorized
        ):
            raise ValueError("Invalid recovery lineage authorization")
        authorized[ref.checkpoint_id] = previous
    generation = claim.generation
    while record.config["configurable"]["checkpoint_id"] != base.checkpoint_id:
        checkpoint_id = record.config["configurable"]["checkpoint_id"]
        if checkpoint_id in authorized:
            previous = authorized[checkpoint_id]
            if previous.generation != generation - 1:
                raise ValueError("Recovery lineage generation mismatch")
            generation = previous.generation
            expected.update(
                app_execution_id=str(previous.execution_id),
                app_generation=previous.generation,
            )
        if checkpoint_id in seen or any(
            record.metadata.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Publication checkpoint lineage mismatch")
        seen.add(checkpoint_id)
        if record.parent_config is None:
            raise ValueError("Publication checkpoint has no accepted ancestor")
        parent = await saver.aget_tuple(record.parent_config)
        if parent is None:
            raise ValueError("Publication ancestor is missing")
        record = parent
