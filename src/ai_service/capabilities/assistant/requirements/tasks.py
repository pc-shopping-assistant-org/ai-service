"""Source-bound task proposals; canonical identities/completion remain application-owned."""

from typing import Annotated, Literal, Protocol

from pydantic import Field, StrictInt, model_validator

from ai_service.capabilities.assistant.stateful_contracts import Contract


class TaskEvidence(Contract):
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    quote: Annotated[str, Field(strict=True, min_length=1, max_length=4000)]

    def validate_source(self, message: str) -> None:
        if (
            self.end <= self.start
            or self.end > len(message)
            or not self.quote.strip()
            or message[self.start : self.end] != self.quote
        ):
            raise ValueError("Task evidence does not match user message")


class TaskDraft(TaskEvidence):
    kind: Literal["BUILD", "COMPARE", "ANSWER", "EXPLAIN_SAVED_BUILD"]
    subjects: list[TaskEvidence] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def comparison_subjects(self) -> "TaskDraft":
        if self.kind == "COMPARE" and (
            len(self.subjects) < 2
            or len({s.quote.casefold() for s in self.subjects}) != len(self.subjects)
        ):
            raise ValueError("Comparison task requires 2..5 distinct product mentions")
        return self

    def validate_source(self, message: str) -> None:
        super().validate_source(message)
        for subject in self.subjects:
            subject.validate_source(message)
            if subject.start < self.start or subject.end > self.end:
                raise ValueError("Task evidence subject is outside task span")


class TaskDraftBatch(Contract):
    """Model proposal only; catalog resolver grants canonical completion targets."""

    tasks: list[TaskDraft] = Field(min_length=1, max_length=8)

    def validate_source(self, message: str) -> None:
        for task in self.tasks:
            task.validate_source(message)


class TaskExtractor(Protocol):
    async def extract_tasks(self, message: str) -> TaskDraftBatch: ...
