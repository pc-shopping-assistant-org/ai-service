import json

import pytest
from test_native_requirement_extractor import Budget, Provider
from test_react_tools import context

from ai_service.capabilities.assistant.react.contracts import DialogueTurn
from ai_service.capabilities.assistant.react.task_resolution import resolve_tasks
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.infrastructure.providers.task_extractor import NativeTaskExtractor


class NoCatalog:
    async def resolve(self, mention):
        raise AssertionError("No product resolution needed for a build task")


@pytest.mark.asyncio
async def test_task_registration_receives_only_safe_published_context():
    trusted = context()
    trusted.history = [
        DialogueTurn(
            user_text="Build PC gaming", assistant_text="Cấu hình đã được đề xuất."
        )
    ]
    message = "Tăng RAM"
    provider = Provider(
        {"tasks": [{"kind": "BUILD", "start": 0, "end": len(message), "quote": message}]}
    )

    await NativeTaskExtractor(provider, Budget(), context=trusted).extract_tasks(message)

    payload = json.loads(provider.model.messages[-1][1])
    assert payload["published_history"][0]["user_text"] == "Build PC gaming"
    assert payload["user_message"] == message
    assert "account_id" not in payload and "catalog" not in payload


@pytest.mark.asyncio
async def test_registration_classifies_sdk_timeout_after_one_reserved_attempt():
    import httpx
    from openai import APITimeoutError

    from ai_service.capabilities.assistant.recovery_policy import RetryFailure
    from ai_service.infrastructure.execution.supervisor import ExecutionFailed

    provider, budget = Provider(None), Budget()

    async def failed(messages):
        raise APITimeoutError(request=httpx.Request("POST", "https://model.test/chat"))

    provider.model.ainvoke = failed
    with pytest.raises(ExecutionFailed) as caught:
        await NativeTaskExtractor(provider, budget).extract_tasks("Build PC")
    assert caught.value.failure == RetryFailure.UPSTREAM_TIMEOUT
    assert len(budget.reservations) == 1


@pytest.mark.asyncio
async def test_task_extractor_returns_task_only_contract_and_validates_source():
    message = "Build PC 25 triệu"
    provider = Provider(
        {"tasks": [{"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}]}
    )

    batch = await NativeTaskExtractor(provider, Budget()).extract_tasks(message)

    assert [task.kind for task in batch.tasks] == ["BUILD"]
    assert not hasattr(batch, "items")
    assert provider.model.schema is TaskDraftBatch


@pytest.mark.asyncio
async def test_task_extractor_rejects_unbound_source_evidence():
    provider = Provider(
        {"tasks": [{"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}]}
    )

    with pytest.raises(ValueError, match="Task evidence"):
        await NativeTaskExtractor(provider, Budget()).extract_tasks("Tìm card đồ họa")


@pytest.mark.asyncio
async def test_comparison_task_preserves_product_mentions_without_generated_ids():
    message = "So RTX 4060 và RTX 4070"
    provider = Provider(
        {
            "tasks": [
                {
                    "kind": "COMPARE",
                    "start": 0,
                    "end": len(message),
                    "quote": message,
                    "subjects": [
                        {"start": 3, "end": 11, "quote": "RTX 4060"},
                        {"start": 15, "end": 23, "quote": "RTX 4070"},
                    ],
                }
            ]
        }
    )

    batch = await NativeTaskExtractor(provider, Budget()).extract_tasks(message)

    assert [subject.quote for subject in batch.tasks[0].subjects] == [
        "RTX 4060",
        "RTX 4070",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subjects",
    [
        [{"start": 3, "end": 11, "quote": "RTX 4060"}],
        [
            {"start": 3, "end": 11, "quote": "RTX 4060"},
            {"start": 15, "end": 23, "quote": "RTX 4090"},
        ],
        [
            {"start": 3, "end": 11, "quote": "RTX 4060", "product_id": "invented"},
            {"start": 15, "end": 23, "quote": "RTX 4070"},
        ],
    ],
)
async def test_comparison_task_rejects_missing_forged_or_id_bearing_subjects(subjects):
    message = "So RTX 4060 và RTX 4070"
    provider = Provider(
        {
            "tasks": [
                {
                    "kind": "COMPARE",
                    "start": 0,
                    "end": len(message),
                    "quote": message,
                    "subjects": subjects,
                }
            ]
        }
    )

    with pytest.raises(ValueError):
        await NativeTaskExtractor(provider, Budget()).extract_tasks(message)


@pytest.mark.asyncio
async def test_task_registration_does_not_extract_or_merge_requirements():
    message = "Build PC 25 triệu"
    provider = Provider(
        {"tasks": [{"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}]}
    )
    budget, state = Budget(), CanonicalRequirementState()
    batch = await NativeTaskExtractor(provider, budget).extract_tasks(message)
    resolved = await resolve_tasks(batch.tasks, message, state, NoCatalog())
    assert [task.kind for task in resolved.requirements] == ["BUILD"]
    assert resolved.requirements[0].constraints_revision == state.revision
    assert not state.requirements
    assert budget.reservations[0][1] == "tasks:register:v2"
    assert "previous_state" not in provider.model.messages[-1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"tasks": []},
        None,
        {
            "tasks": [{"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}],
            "items": [],
        },
        {"tasks": [{"kind": "BUILD", "start": 0, "end": 8, "quote": "Invented"}]},
    ],
)
async def test_invalid_registration_never_becomes_default_answer(result):
    provider = Provider(result)
    with pytest.raises(ValueError):
        await NativeTaskExtractor(provider, Budget()).extract_tasks("Build PC")


@pytest.mark.asyncio
async def test_task_registration_budget_rejection_prevents_model_dispatch():
    provider = Provider({"tasks": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeTaskExtractor(provider, Budget(denied=True)).extract_tasks(
            "GPU?"
        )
    assert provider.model.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_registration_compares_canonical_products_or_keeps_unresolved(ambiguous):
    from test_task_resolution import Catalog, compare_task, product

    task = compare_task()
    provider = Provider({"tasks": [task.model_dump(mode="json")]})
    batch = await NativeTaskExtractor(provider, Budget()).extract_tasks(
        task.quote
    )
    first, second = product("ASUS RTX 4060"), product("MSI RTX 4070")
    result = await resolve_tasks(
        batch.tasks,
        task.quote,
        CanonicalRequirementState(),
        Catalog(
            {
                "RTX 4060": [first, product("MSI RTX 4060")] if ambiguous else [first],
                "RTX 4070": [second],
            }
        ),
    )
    if ambiguous:
        assert not result.requirements
        assert result.pending[0].reason == "AMBIGUOUS"
    else:
        assert result.requirements[0].kind == "COMPARE"
        assert result.requirements[0].product_ids == [
            first.product_id,
            second.product_id,
        ]


@pytest.mark.asyncio
async def test_invalid_user_message_does_not_reserve_or_dispatch():
    budget, provider = Budget(), Provider(None)
    with pytest.raises(ValueError):
        await NativeTaskExtractor(provider, budget).extract_tasks(" " * 5)
    assert not budget.reservations
    assert not provider.model.messages
