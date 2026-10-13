"""Versioned typed requirements; no expression parser or optimizer-field aliases."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StrictInt, model_validator

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    QuantityUnit,
    Text,
)

Subject = Literal["SETUP", "WORKLOAD"] | ComponentCategory


class SourceEvidence(Contract):
    message_id: UUID
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    quote: Text

    @model_validator(mode="after")
    def span_order(self) -> "SourceEvidence":
        if self.end <= self.start:
            raise ValueError("Invalid evidence span")
        return self


class Evidence(SourceEvidence):
    # Absent supporting evidence must not change hashes of existing snapshots.
    supporting_evidence: list[SourceEvidence] = Field(
        default_factory=list, max_length=4, exclude_if=lambda value: not value
    )


class QuantityValue(Contract):
    kind: Literal["QUANTITY"] = "QUANTITY"
    amount: StrictInt = Field(ge=0)
    unit: QuantityUnit


class TextValue(Contract):
    kind: Literal["TEXT"] = "TEXT"
    text: Text


class ReferenceValue(Contract):
    kind: Literal["REFERENCE"] = "REFERENCE"
    variant_id: UUID


Value = Annotated[
    QuantityValue | TextValue | ReferenceValue, Field(discriminator="kind")
]


class Requirement(Contract):
    id: UUID
    subject: Subject
    predicate: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    operator: Literal["EQ", "GTE", "LTE"] = "EQ"
    value: Value
    strength: Literal["HARD", "SOFT"] = "HARD"
    evidence: Evidence


class Resource(Contract):
    id: UUID
    variant_id: UUID
    evidence: Evidence


class Binding(Contract):
    requirement_id: UUID
    variant_id: UUID
    evidence_refs: list[Text] = Field(min_length=1, max_length=20)


class CanonicalRequirementState(Contract):
    schema_version: Literal[1] = 1
    revision: StrictInt = Field(default=0, ge=0)
    goals: dict[UUID, Requirement] = Field(default_factory=dict, max_length=50)
    requirements: dict[UUID, Requirement] = Field(default_factory=dict, max_length=100)
    available_resources: dict[UUID, Resource] = Field(
        default_factory=dict, max_length=30
    )
    resolved_bindings: dict[UUID, Binding] = Field(default_factory=dict, max_length=100)
    unresolved: list[Text] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def identities(self) -> "CanonicalRequirementState":
        for collection in (self.goals, self.requirements, self.available_resources):
            if any(key != item.id for key, item in collection.items()):
                raise ValueError("State key must match item identity")
        all_ids = [*self.goals, *self.requirements, *self.available_resources]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("State identities must be unique")
        for key, binding in self.resolved_bindings.items():
            if key != binding.requirement_id or key not in self.requirements:
                raise ValueError("Binding must reference an existing requirement")
        return self


class RequirementPatch(Contract):
    set_requirements: list[Requirement] = Field(default_factory=list, max_length=100)
    clear_requirements: list[UUID] = Field(default_factory=list, max_length=100)
    set_goals: list[Requirement] = Field(default_factory=list, max_length=50)
    clear_goals: list[UUID] = Field(default_factory=list, max_length=50)
    set_resources: list[Resource] = Field(default_factory=list, max_length=30)
    clear_resources: list[UUID] = Field(default_factory=list, max_length=30)
    clear_evidence: dict[UUID, Evidence] = Field(default_factory=dict, max_length=180)

    @model_validator(mode="after")
    def unique_operations(self) -> "RequirementPatch":
        all_operations = [
            *(item.id for item in self.set_requirements),
            *self.clear_requirements,
            *(item.id for item in self.set_goals),
            *self.clear_goals,
            *(item.id for item in self.set_resources),
            *self.clear_resources,
        ]
        if len(all_operations) != len(set(all_operations)):
            raise ValueError("Duplicate patch identity across operations")
        cleared = set(self.clear_requirements + self.clear_goals + self.clear_resources)
        if cleared != set(self.clear_evidence):
            raise ValueError("Every CLEAR requires bound source evidence")
        for setting, clearing in (
            (self.set_requirements, self.clear_requirements),
            (self.set_goals, self.clear_goals),
            (self.set_resources, self.clear_resources),
        ):
            ids = [item.id for item in setting]
            if len(ids) != len(set(ids)) or len(clearing) != len(set(clearing)):
                raise ValueError("Duplicate patch operation")
            if set(ids) & set(clearing):
                raise ValueError("Cannot SET and CLEAR the same identity")
        return self


def requirements_hash(state: CanonicalRequirementState) -> str:
    """V1 content identity includes semantics/IDs/evidence, not the legacy counter."""
    checked = CanonicalRequirementState.model_validate(state.model_dump())
    return canonical_hash(
        {
            "hash_version": "requirements-content:v1",
            "state": checked.model_dump(mode="json", exclude={"revision"}),
        }
    )
