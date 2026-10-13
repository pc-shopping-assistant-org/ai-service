from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ai_service.capabilities.assistant.react.contracts import ProductEvidence
from ai_service.capabilities.assistant.react.task_resolution import resolve_tasks
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraft


class Catalog:
    def __init__(self, matches):
        self.matches = matches

    async def resolve(self, mention):
        return self.matches.get(mention, [])


def product(name):
    return ProductEvidence(
        product_id=uuid4(),
        name=name,
        evidence_ref=f"catalog:{name}",
        retrieved_at=datetime.now(UTC),
    )


def compare_task():
    return TaskDraft(
        kind="COMPARE",
        start=0,
        end=23,
        quote="So RTX 4060 và RTX 4070",
        subjects=[
            {"start": 3, "end": 11, "quote": "RTX 4060"},
            {"start": 15, "end": 23, "quote": "RTX 4070"},
        ],
    )


@pytest.mark.asyncio
async def test_resolved_comparison_is_bound_to_canonical_product_ids():
    first, second = product("ASUS RTX 4060"), product("MSI RTX 4070")
    result = await resolve_tasks(
        [compare_task()],
        "So RTX 4060 và RTX 4070",
        CanonicalRequirementState(),
        Catalog({"RTX 4060": [first], "RTX 4070": [second]}),
    )
    assert not result.pending
    assert result.requirements[0].kind == "COMPARE"
    assert result.requirements[0].product_ids == [first.product_id, second.product_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("found", [False, True])
async def test_missing_or_ambiguous_product_blocks_whole_task_set(found):
    first, second = product("ASUS RTX 4060"), product("MSI RTX 4060")
    result = await resolve_tasks(
        [
            TaskDraft(kind="ANSWER", start=0, end=23, quote="So RTX 4060 và RTX 4070"),
            compare_task(),
        ],
        "So RTX 4060 và RTX 4070",
        CanonicalRequirementState(),
        Catalog({"RTX 4060": [first, second] if found else []}),
    )
    assert result.requirements == []
    assert result.pending[0].reason == ("AMBIGUOUS" if found else "NOT_FOUND")
    assert result.pending[0].subject_index == 0
    assert len(result.pending[0].candidates) == (2 if found else 0)


@pytest.mark.asyncio
async def test_empty_task_draft_never_defaults_to_build():
    with pytest.raises(ValueError, match="TASK_DRAFT_REQUIRED"):
        await resolve_tasks([], "Hello", CanonicalRequirementState(), Catalog({}))


@pytest.mark.asyncio
async def test_build_task_uses_server_requirements_revision():
    state = CanonicalRequirementState(revision=4)
    result = await resolve_tasks(
        [TaskDraft(kind="BUILD", start=0, end=8, quote="Build PC")],
        "Build PC",
        state,
        Catalog({}),
    )
    assert result.requirements[0].constraints_revision == 4


@pytest.mark.asyncio
async def test_saved_build_task_binds_only_application_supplied_snapshot():
    message = "Giải thích cấu hình cũ"
    task = TaskDraft(
        kind="EXPLAIN_SAVED_BUILD", start=0, end=len(message), quote=message
    )
    result = await resolve_tasks(
        [task],
        message,
        CanonicalRequirementState(),
        Catalog({}),
        saved_build_ref="saved-build:accepted",
    )
    assert not result.pending
    assert result.requirements[0].saved_build_ref == "saved-build:accepted"
    missing = await resolve_tasks(
        [task], message, CanonicalRequirementState(), Catalog({})
    )
    assert missing.requirements == []
    assert missing.pending[0].reason == "SNAPSHOT_REQUIRED"
