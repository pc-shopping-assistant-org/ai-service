"""Small native tool inputs. Application DTOs remain the validation authority."""

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictInt

from ai_service.capabilities.assistant.requirements.contracts import (
    QuantityValue,
    Subject,
    TextValue,
)
from ai_service.capabilities.assistant.source_quotes import (
    QuotedDraftItem,
    QuotedRequirementDraft,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    QuantityUnit,
    Text,
)

from .contracts import ProductReview, WebExcerpt
from .submissions import ResultSelection, SubmitResults


class RequirementInput(Contract):
    subject: Subject
    predicate: Text
    value: StrictInt | str | None = None
    unit: QuantityUnit | None = None
    quote: Text
    target: Literal["REQUIREMENT", "GOAL", "RESOURCE"] = "REQUIREMENT"
    operation: Literal["SET", "CLEAR"] = "SET"
    target_id: UUID | None = None
    operator: Literal["EQ", "GTE", "LTE"] = "EQ"
    strength: Literal["HARD", "SOFT"] = "HARD"

    def quoted(self) -> QuotedDraftItem:
        if self.unit is not None and not isinstance(self.value, int):
            raise ValueError("Only a numeric quantity can have a unit")
        value = (
            {"kind": "QUANTITY", "amount": self.value, "unit": self.unit}
            if self.unit is not None
            else {"kind": "UNRESOLVED_QUANTITY", "amount": self.value}
            if isinstance(self.value, int)
            else {"kind": "TEXT", "text": self.value}
            if self.value is not None
            else None
        )
        return QuotedDraftItem.model_validate(
            self.model_dump(exclude={"value", "unit"}) | {"value": value}
        )

    @classmethod
    def from_quoted(cls, item: QuotedDraftItem) -> "RequirementInput":
        value = item.value
        return cls.model_validate(
            item.model_dump(exclude={"value"})
            | {
                "value": value.text
                if isinstance(value, TextValue)
                else value.amount
                if value
                else None,
                "unit": value.unit if isinstance(value, QuantityValue) else None,
            }
        )


class RequirementProposalInput(Contract):
    items: list[RequirementInput] = Field(min_length=1, max_length=100)
    source_text: Text | None = Field(default=None, max_length=4000)

    def quoted(self) -> QuotedRequirementDraft:
        return QuotedRequirementDraft(items=[item.quoted() for item in self.items])


class EvidenceResultInput(Contract):
    text: Text = Field(max_length=8000)
    refs: list[Text] = Field(default_factory=list, max_length=100)
    build_ref: UUID | None = None
    calls: list[Text] = Field(default_factory=list, max_length=20)
    task_index: StrictInt | None = Field(default=0, ge=0, lt=8)
    explanation: Text | None = Field(default=None, max_length=3000)
    explanation_refs: list[Text] | None = Field(default=None, max_length=20)


class ResultInput(EvidenceResultInput):
    quotes: list[WebExcerpt] = Field(default_factory=list, max_length=8)
    review: ProductReview | None = None

    def selection(self) -> ResultSelection:
        return ResultSelection(
            text=self.text,
            evidence_refs=self.refs,
            build_ref=self.build_ref,
            supporting_tool_call_ids=self.calls,
            task_index=self.task_index,
            web_excerpts=self.quotes,
            explanation=self.explanation,
            explanation_evidence_refs=self.explanation_refs,
            review=self.review,
        )

    @classmethod
    def from_selection(cls, selected: ResultSelection) -> "ResultInput":
        return cls(
            text=selected.text,
            refs=selected.evidence_refs,
            build_ref=selected.build_ref,
            calls=selected.supporting_tool_call_ids,
            task_index=selected.task_index,
            quotes=selected.web_excerpts,
            explanation=selected.explanation,
            explanation_refs=selected.explanation_evidence_refs,
            review=selected.review,
        )


class ResultSubmissionInput(Contract):
    results: list[ResultInput] = Field(min_length=1, max_length=8)

    def submission(self) -> SubmitResults:
        return SubmitResults(results=[item.selection() for item in self.results])


class EvidenceSubmissionInput(Contract):
    """Catalog/build submissions need no web quotation or review fields."""

    results: list[EvidenceResultInput] = Field(min_length=1, max_length=8)
