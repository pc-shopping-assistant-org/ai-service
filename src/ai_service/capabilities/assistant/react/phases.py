"""ReAct transitions shared by graph nodes, routing and publication checks."""

from enum import StrEnum
from typing import NewType

ReActPhaseCode = NewType("ReActPhaseCode", str)


class ReActPhase(StrEnum):
    STARTED = "STARTED"
    REGISTER_REQUEST = "REGISTER_REQUEST"
    INTERPRET_REPLY = "INTERPRET_REPLY"
    AMEND_PENDING = "AMEND_PENDING"
    AMEND_TASK = "AMEND_TASK"
    DECIDED = "DECIDED"
    REPAIR = "REPAIR"
    TOOL_PENDING = "TOOL_PENDING"
    TOOL_COMMITTED = "TOOL_COMMITTED"
    OBSERVED = "OBSERVED"
    RESOLVE_PROPOSAL = "RESOLVE_PROPOSAL"
    REQUIREMENTS_UNCHANGED = "REQUIREMENTS_UNCHANGED"
    REQUIREMENTS_ACCEPTED = "REQUIREMENTS_ACCEPTED"
    WAITING_INPUT = "WAITING_INPUT"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"

    @property
    def code(self) -> ReActPhaseCode:
        """Store a plain string so native checkpoints retain their wire format."""
        return ReActPhaseCode(self.value)
