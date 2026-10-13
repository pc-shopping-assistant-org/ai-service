"""Trusted optimizer report shape. Metadata validation is not itself a search proof.

Only application/optimizer adapters may produce this evidence. F03/B03 must derive
completeness/coverage/pruning guarantees from actual execution and test the algorithm.
"""

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, model_validator

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text


class BuildSearchEvidence(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    outcome: Literal["FEASIBLE", "INFEASIBLE", "SEARCH_INCOMPLETE"]
    scope: Text = Field(max_length=1000)
    requirements_hash: Digest
    optimizer_input_hash: Digest
    catalog_snapshot_hash: Digest
    algorithm_version: Text
    pruning_version: Text
    policy_version: Text
    search_complete: StrictBool
    coverage_complete: StrictBool
    pruning_sound: StrictBool
    termination: Literal[
        "FOUND_RESULT", "EXHAUSTED", "TIME_LIMIT", "BUDGET_LIMIT", "HEURISTIC_STOP"
    ]
    build_ref: UUID | None = None

    @model_validator(mode="after")
    def scoped_outcome(self) -> "BuildSearchEvidence":
        if (self.termination == "FOUND_RESULT") != (self.outcome == "FEASIBLE"):
            raise ValueError("Found-result termination must match feasible outcome")
        if self.outcome == "FEASIBLE":
            if self.build_ref is None:
                raise ValueError(
                    "Feasible outcome requires a verifier-bound build result"
                )
        elif self.build_ref is not None:
            raise ValueError("No-solution outcome cannot contain a feasible build")
        declares_complete_search = (
            self.search_complete
            and self.coverage_complete
            and self.pruning_sound
            and self.termination == "EXHAUSTED"
        )
        if self.outcome == "INFEASIBLE" and not declares_complete_search:
            raise ValueError("Infeasible outcome requires scoped completeness proof")
        if self.outcome == "SEARCH_INCOMPLETE" and declares_complete_search:
            raise ValueError("Complete exhaustion proof is not an incomplete search")
        return self
