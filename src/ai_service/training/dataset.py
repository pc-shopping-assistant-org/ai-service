"""Validate versioned pilot labels against existing runtime contracts, without I/O tools."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.limits import MAX_TURN_MESSAGE_CHARS
from ai_service.capabilities.assistant.react.actions import (
    CallToolAction,
    FinalAnswerAction,
    FlexibleDecision,
    ProposeRequirementsAction,
)
from ai_service.capabilities.assistant.react.contracts import (
    TOOL_SCHEMAS,
    BuildPCArgs,
    CompletionRequirement,
    ToolContext,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
from ai_service.capabilities.assistant.react.tools import (
    allowed_tools,
    project_current_tools,
)
from ai_service.capabilities.assistant.requirements.contracts import QuantityValue
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.requirements.units import verify_quantity
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

SAMPLES_FILE = "samples.jsonl"
SNAPSHOTS_FILE = "snapshots.json"
MANIFEST_FILE = "manifest.json"
# Preserve the original text: stripping/normalizing it would invalidate source offsets.
type SourceMessage = Annotated[
    str, Field(strict=True, min_length=1, max_length=MAX_TURN_MESSAGE_CHARS)
]


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ReviewStatus(StrEnum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"


class DatasetTask(StrEnum):
    TASK_EXTRACTION = "TASK_EXTRACTION"
    REQUIREMENT_EXTRACTION = "REQUIREMENT_EXTRACTION"
    REACT_DECISION = "REACT_DECISION"


class DatasetManifest(Contract):
    schema_version: Literal[1] = 1
    dataset_id: Text
    runtime_contract: Literal["react-v2"]
    source: Literal["AUTHORED_SYNTHETIC"]
    files: dict[str, Digest]

    @model_validator(mode="after")
    def expected_files(self) -> "DatasetManifest":
        if set(self.files) != {SAMPLES_FILE, SNAPSHOTS_FILE}:
            raise ValueError("DATASET_FILES_INVALID")
        return self


class DatasetSnapshot(Contract):
    context: ToolContext
    records: list[ToolRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def committed_current_results(self) -> "DatasetSnapshot":
        if len(project_current_tools(self.records, self.context)) != len(self.records):
            raise ValueError("SNAPSHOT_STALE_RESULTS")
        for requirement in (
            *self.context.requirements.requirements.values(),
            *self.context.requirements.goals.values(),
        ):
            if isinstance(requirement.value, QuantityValue):
                normalized, reason = verify_quantity(
                    requirement.evidence.quote, requirement.value
                )
                if normalized is None:
                    raise ValueError(reason)
        return self


class SampleMetadata(Contract):
    sample_id: Text
    # A scenario family includes all turns/paraphrases; it must stay in one split.
    scenario_id: Text
    split: DatasetSplit
    snapshot_id: Text
    message: SourceMessage
    expected_behavior: Text
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewed_by: Text | None = None

    @model_validator(mode="after")
    def explicit_review(self) -> "SampleMetadata":
        if (self.review_status == ReviewStatus.APPROVED) != (
            self.reviewed_by is not None
        ):
            raise ValueError("REVIEW_IDENTITY_REQUIRED")
        return self


class TaskExtractionSample(SampleMetadata):
    task: Literal[DatasetTask.TASK_EXTRACTION]
    target: TaskDraftBatch


class RequirementExtractionSample(SampleMetadata):
    task: Literal[DatasetTask.REQUIREMENT_EXTRACTION]
    target: RequirementDraft


class ReactDecisionSample(SampleMetadata):
    task: Literal[DatasetTask.REACT_DECISION]
    target: FlexibleDecision


type TrainingSample = Annotated[
    TaskExtractionSample | RequirementExtractionSample | ReactDecisionSample,
    Field(discriminator="task"),
]
SAMPLE_ADAPTER: TypeAdapter[TrainingSample] = TypeAdapter(TrainingSample)
SNAPSHOT_ADAPTER = TypeAdapter(dict[str, DatasetSnapshot])


class DatasetReport(Contract):
    dataset_id: str
    samples: int
    scenarios: int
    tasks: dict[str, int]
    splits: dict[str, int]
    review_status: dict[str, int]
    train_ready: bool


def validate_requirement_draft(
    draft: RequirementDraft, source: str, context: ToolContext
) -> None:
    for item in draft.items:
        if (
            not 0 <= item.start < item.end <= len(source)
            or not source[item.start : item.end].strip()
        ):
            raise ValueError("SOURCE_SPAN_INVALID")
        if isinstance(item.value, QuantityValue):
            normalized, reason = verify_quantity(
                source[item.start : item.end], item.value
            )
            if normalized is None:
                raise ValueError(reason)
        existing = {
            "REQUIREMENT": context.requirements.requirements.keys(),
            "GOAL": context.requirements.goals.keys(),
            "RESOURCE": context.requirements.available_resources.keys(),
        }
        if item.target_id is not None and item.target_id not in existing[item.target]:
            raise ValueError("TARGET_NOT_IN_SNAPSHOT")
        if item.operation == "CLEAR" and (
            item.target_id is None or item.value is not None
        ):
            raise ValueError("CLEAR_TARGET_INVALID")


def validate_label(sample: TrainingSample, snapshot: DatasetSnapshot) -> None:
    """Mechanical checks only. A valid span/schema does not prove semantic intent."""
    if isinstance(sample, TaskExtractionSample):
        sample.target.validate_source(sample.message)
    elif isinstance(sample, RequirementExtractionSample):
        validate_requirement_draft(sample.target, sample.message, snapshot.context)
    else:
        action = sample.target.action
        if isinstance(action, CallToolAction):
            name = ToolName(action.call.name)
            if name not in allowed_tools(snapshot.context, snapshot.records):
                raise ValueError("TARGET_TOOL_NOT_ALLOWED")
            arguments = TOOL_SCHEMAS[name].model_validate(action.call.arguments)
            if isinstance(arguments, BuildPCArgs) and arguments.objective is not None:
                raise ValueError("UNSUPPORTED_BUILD_OBJECTIVE")
        elif isinstance(action, ProposeRequirementsAction):
            validate_requirement_draft(
                action.draft, action.source_text or sample.message, snapshot.context
            )
        elif isinstance(action, FinalAnswerAction):
            validate_final_answer(
                action,
                [record.result for record in snapshot.records if record.result],
                snapshot.context,
                snapshot.context.completion_requirements
                or [CompletionRequirement(kind="ANSWER")],
                PCBuildApplicationService(),
            )


@dataclass
class LoadedDataset:
    manifest: DatasetManifest
    snapshots: dict[str, DatasetSnapshot]
    samples: list[TrainingSample]
    report: DatasetReport


def load_dataset(directory: Path, *, require_approved: bool = False) -> LoadedDataset:
    manifest = DatasetManifest.model_validate_json(
        (directory / MANIFEST_FILE).read_bytes()
    )
    files = {name: (directory / name).read_bytes() for name in manifest.files}
    for name, content in files.items():
        if hashlib.sha256(content).hexdigest() != manifest.files[name]:
            raise ValueError(f"CONTENT_HASH_MISMATCH: {name}")
    snapshots = SNAPSHOT_ADAPTER.validate_json(files[SNAPSHOTS_FILE])
    samples: list[TrainingSample] = []
    scenario_splits: dict[str, DatasetSplit] = {}
    identifiers: set[str] = set()
    for line_number, line in enumerate(files[SAMPLES_FILE].splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = SAMPLE_ADAPTER.validate_json(line)
            if sample.sample_id in identifiers:
                raise ValueError("DUPLICATE_SAMPLE")
            identifiers.add(sample.sample_id)
            previous = scenario_splits.setdefault(sample.scenario_id, sample.split)
            if previous != sample.split:
                raise ValueError("SCENARIO_SPLIT_LEAK")
            if sample.snapshot_id not in snapshots:
                raise ValueError("SNAPSHOT_NOT_FOUND")
            validate_label(sample, snapshots[sample.snapshot_id])
            samples.append(sample)
        except ValueError as error:
            raise ValueError(f"{SAMPLES_FILE}:{line_number}: {error}") from error
    if not samples:
        raise ValueError("DATASET_EMPTY")
    approved = all(sample.review_status == ReviewStatus.APPROVED for sample in samples)
    if require_approved and not approved:
        raise ValueError("DATASET_REVIEW_REQUIRED")
    report = DatasetReport(
        dataset_id=manifest.dataset_id,
        samples=len(samples),
        scenarios=len(scenario_splits),
        tasks=dict(Counter(sample.task for sample in samples)),
        splits=dict(Counter(sample.split for sample in samples)),
        review_status=dict(Counter(sample.review_status for sample in samples)),
        train_ready=approved
        and {sample.split for sample in samples} == set(DatasetSplit),
    )
    return LoadedDataset(manifest, snapshots, samples, report)


def validate_dataset(
    directory: Path, *, require_approved: bool = False
) -> DatasetReport:
    return load_dataset(directory, require_approved=require_approved).report
