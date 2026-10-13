"""Explicit whole-pending actions do not clear accepted canonical requirements."""

from typing import Literal

from pydantic import Field

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.task_resolution import PendingTaskDraft
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    PendingActionV1,
)

from .contracts import CanonicalRequirementState
from .resolver import PendingDraft

DISCARDED_OUTCOME = "PENDING_DISCARDED"
DISCARDED_ANSWER = "Đã bỏ yêu cầu đang chờ làm rõ. Yêu cầu và cấu hình đã lưu được giữ nguyên."


class PendingActionReceipt(Contract):
    action: PendingActionV1
    pending_kind: Literal["REQUIREMENTS", "TASKS"]
    pending_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def validate_pending_action(
    action: PendingActionV1,
    state: CanonicalRequirementState,
    pending: PendingDraft | None,
    pending_tasks: PendingTaskDraft | None,
) -> PendingActionReceipt:
    action = PendingActionV1.model_validate(action.model_dump())
    if (pending is None) == (pending_tasks is None):
        raise ValueError("Exactly one pending artifact is required")
    artifact = pending if pending is not None else pending_tasks
    assert artifact is not None
    artifact = type(artifact).model_validate(artifact.model_dump())
    if action.pending_message_id != artifact.message_id:
        raise ValueError("Pending action source mismatch")
    if action.constraints_revision != artifact.constraints_revision or action.constraints_revision != state.revision:
        raise ValueError("Stale pending action revision")
    return PendingActionReceipt(
        action=action,
        pending_kind="REQUIREMENTS" if pending is not None else "TASKS",
        pending_hash=canonical_hash(artifact),
        requirements_hash=canonical_hash(state),
    )
