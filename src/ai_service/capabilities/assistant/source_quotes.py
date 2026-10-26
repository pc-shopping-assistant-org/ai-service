"""Model-facing quotes; source offsets belong to application code only."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from ai_service.capabilities.assistant.react.task_amendment import TaskAmendment
from ai_service.capabilities.assistant.react.task_resolution import TaskChoiceDraft
from ai_service.capabilities.assistant.requirements.amendment import (
    PendingAmendment,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    QuantityValue,
    Subject,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitChoiceDraft,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    ChoiceDraft,
    RequirementDraft,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.requirements.units import UnresolvedQuantityValue
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    QuantityUnit,
)


def quote_span(
    source: str, quote: str, *, start: int = 0, end: int | None = None
) -> dict[str, int]:
    """Bind one exact occurrence, using Python character offsets like source slicing."""
    if not quote or not quote.strip():
        raise ValueError("Empty source quote")
    bound = len(source) if end is None else end
    position = source.find(quote, start, bound)
    if position < 0:
        raise ValueError("Source quote not found")
    if source.find(quote, position + 1, bound) >= 0:
        raise ValueError("Ambiguous source quote")
    return {"start": position, "end": position + len(quote)}


class QuotedEvidence(Contract):
    quote: str = Field(min_length=1, max_length=4000)

    def bound(
        self, source: str, *, start: int = 0, end: int | None = None
    ) -> dict[str, str | int]:
        return {
            "quote": self.quote,
            **quote_span(source, self.quote, start=start, end=end),
        }


class QuotedTask(QuotedEvidence):
    kind: Literal["BUILD", "COMPARE", "ANSWER", "EXPLAIN_SAVED_BUILD"]
    subjects: list[QuotedEvidence] = Field(default_factory=list, max_length=5)


class QuotedTaskBatch(Contract):
    tasks: list[QuotedTask] = Field(max_length=8)

    def bind(self, source: str) -> TaskDraftBatch:
        tasks = []
        for task in self.tasks:
            task_span = quote_span(source, task.quote)
            tasks.append(
                {
                    "kind": task.kind,
                    "quote": task.quote,
                    **task_span,
                    "subjects": [
                        subject.bound(source, **task_span) for subject in task.subjects
                    ],
                }
            )
        batch = TaskDraftBatch.model_validate({"tasks": tasks})
        batch.validate_source(source)
        return batch


class QuotedDraftItem(Contract):
    target: Literal["REQUIREMENT", "GOAL", "RESOURCE"] = "REQUIREMENT"
    operation: Literal["SET", "CLEAR"] = "SET"
    target_id: UUID | None = None
    subject: Subject
    predicate: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    operator: Literal["EQ", "GTE", "LTE"] = "EQ"
    strength: Literal["HARD", "SOFT"] = "HARD"
    value: (
        Annotated[
            QuantityValue | UnresolvedQuantityValue | TextValue,
            Field(discriminator="kind"),
        ]
        | None
    ) = None
    quote: str = Field(min_length=1, max_length=4000)

    def bound(self, source: str) -> dict[str, object]:
        return {
            **self.model_dump(exclude={"quote"}),
            **quote_span(source, self.quote),
        }


class QuotedRequirementDraft(Contract):
    schema_version: Literal[2] = 2
    items: list[QuotedDraftItem] = Field(max_length=100)

    def bind(self, source: str) -> RequirementDraft:
        return RequirementDraft.model_validate(
            {"items": [item.bound(source) for item in self.items]}
        )


class QuotedReplacement(Contract):
    item_index: NonNegativeInt
    item: QuotedDraftItem


class QuotedDiscard(QuotedEvidence):
    item_index: NonNegativeInt


class QuotedPendingAmendment(Contract):
    replacements: list[QuotedReplacement] = Field(default_factory=list, max_length=100)
    discards: list[QuotedDiscard] = Field(default_factory=list, max_length=100)

    def bind(self, source: str) -> PendingAmendment:
        return PendingAmendment.model_validate(
            {
                "replacements": [
                    {"item_index": item.item_index, "item": item.item.bound(source)}
                    for item in self.replacements
                ],
                "discards": [
                    {"item_index": item.item_index, **item.bound(source)}
                    for item in self.discards
                ],
            }
        )


class QuotedUnitSelection(QuotedEvidence):
    item_index: NonNegativeInt
    unit: QuantityUnit


class QuotedUnitChoice(Contract):
    selections: list[QuotedUnitSelection] = Field(default_factory=list, max_length=100)

    def bind(self, source: str) -> UnitChoiceDraft:
        return UnitChoiceDraft.model_validate(
            {
                "selections": [
                    {
                        "item_index": item.item_index,
                        "unit": item.unit,
                        **item.bound(source),
                    }
                    for item in self.selections
                ]
            }
        )


class QuotedChoiceSelection(QuotedEvidence):
    item_index: NonNegativeInt
    variant_id: UUID


class QuotedChoice(Contract):
    selections: list[QuotedChoiceSelection] = Field(
        default_factory=list, max_length=100
    )

    def bind(self, source: str) -> ChoiceDraft:
        return ChoiceDraft.model_validate(
            {
                "selections": [
                    {
                        "item_index": item.item_index,
                        "variant_id": item.variant_id,
                        **item.bound(source),
                    }
                    for item in self.selections
                ]
            }
        )


class QuotedTaskChoiceSelection(Contract):
    task_index: NonNegativeInt
    subject_index: NonNegativeInt
    product_id: UUID
    evidence: QuotedEvidence


class QuotedTaskChoice(Contract):
    selections: list[QuotedTaskChoiceSelection] = Field(
        default_factory=list, max_length=40
    )

    def bind(self, source: str) -> TaskChoiceDraft:
        return TaskChoiceDraft.model_validate(
            {
                "selections": [
                    {
                        "task_index": item.task_index,
                        "subject_index": item.subject_index,
                        "product_id": item.product_id,
                        "evidence": item.evidence.bound(source),
                    }
                    for item in self.selections
                ]
            }
        )


class QuotedTaskMentionEdit(Contract):
    task_index: NonNegativeInt
    subject_index: NonNegativeInt
    evidence: QuotedEvidence


class QuotedTaskAmendment(Contract):
    edits: list[QuotedTaskMentionEdit] = Field(default_factory=list, max_length=40)

    def bind(self, source: str) -> TaskAmendment:
        return TaskAmendment.model_validate(
            {
                "edits": [
                    {
                        "task_index": item.task_index,
                        "subject_index": item.subject_index,
                        "evidence": item.evidence.bound(source),
                    }
                    for item in self.edits
                ]
            }
        )


def project_requirement_draft(
    draft: RequirementDraft, source: str
) -> QuotedRequirementDraft:
    """Project reviewed internal labels into the serving model contract."""
    return QuotedRequirementDraft.model_validate(
        {
            "items": [
                {
                    **item.model_dump(exclude={"start", "end"}),
                    "quote": source[item.start : item.end],
                }
                for item in draft.items
            ]
        }
    )


def project_task_batch(batch: TaskDraftBatch, source: str) -> QuotedTaskBatch:
    batch.validate_source(source)
    return QuotedTaskBatch.model_validate(
        {
            "tasks": [
                {
                    "kind": task.kind,
                    "quote": task.quote,
                    "subjects": [{"quote": subject.quote} for subject in task.subjects],
                }
                for task in batch.tasks
            ]
        }
    )
