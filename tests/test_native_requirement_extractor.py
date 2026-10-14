import json

import pytest
from pydantic import ValidationError
from test_react_tools import context

from ai_service.capabilities.assistant.react.contracts import DialogueTurn
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)


@pytest.mark.asyncio
async def test_task_registration_receives_published_context_for_follow_up():
    trusted = context()
    trusted.history = [
        DialogueTurn(
            user_text="Build PC gaming", assistant_text="Cấu hình đã được đề xuất."
        )
    ]
    message = "Tăng RAM"
    provider = Provider(
        {
            "tasks": [
                {"kind": "BUILD", "start": 0, "end": len(message), "quote": message}
            ]
        }
    )
    await NativeRequirementExtractor(provider, Budget(), context=trusted).extract_tasks(
        message
    )
    payload = json.loads(provider.model.messages[-1][1])
    assert payload["published_history"][0]["user_text"] == "Build PC gaming"
    assert payload["user_message"] == message
    assert "account_id" not in payload and "catalog" not in payload


class Model:
    def __init__(self, result):
        self.result = result
        self.messages = []

    def with_structured_output(self, schema, **kwargs):
        self.schema, self.options = schema, kwargs
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.result


class Provider:
    name = "openai"

    def __init__(self, result):
        self.model = Model(result)

    def create_model(self):
        return self.model


class Budget:
    def __init__(self, denied=False):
        self.denied = denied
        self.reservations = []

    async def invoke(self, kind, key, digest, dispatch):
        self.reservations.append((kind, key, digest))
        if self.denied:
            raise ValueError("CALL_BUDGET_EXHAUSTED")
        return await dispatch()


@pytest.mark.asyncio
async def test_extraction_keeps_build_task_separate_from_requirements():
    message = "Build PC 25 triệu"
    task = {"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}
    provider = Provider({"items": [], "tasks": [task]})
    result = await NativeRequirementExtractor(provider, Budget()).extract(
        message, CanonicalRequirementState()
    )
    assert result.tasks[0].kind == "BUILD"
    assert result.tasks[0].quote == "Build PC"
    assert result.items == []


@pytest.mark.asyncio
async def test_extraction_rejects_unbound_task_evidence():
    provider = Provider(
        {
            "items": [],
            "tasks": [{"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}],
        }
    )
    with pytest.raises(ValueError, match="Task evidence"):
        await NativeRequirementExtractor(provider, Budget()).extract(
            "Tìm card đồ họa", CanonicalRequirementState()
        )


@pytest.mark.asyncio
async def test_comparison_task_preserves_mentions_without_model_generated_ids():
    message = "So RTX 4060 và RTX 4070"
    provider = Provider(
        {
            "items": [],
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
            ],
        }
    )
    result = await NativeRequirementExtractor(provider, Budget()).extract(
        message, CanonicalRequirementState()
    )
    assert [s.quote for s in result.tasks[0].subjects] == ["RTX 4060", "RTX 4070"]


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
            "items": [],
            "tasks": [
                {
                    "kind": "COMPARE",
                    "start": 0,
                    "end": len(message),
                    "quote": message,
                    "subjects": subjects,
                }
            ],
        }
    )
    with pytest.raises(ValueError):
        await NativeRequirementExtractor(provider, Budget()).extract(
            message, CanonicalRequirementState()
        )


@pytest.mark.asyncio
async def test_native_extraction_uses_typed_draft_and_durable_model_boundary():
    message = "Build 25 triệu"
    provider = Provider(
        {
            "items": [
                {
                    "subject": "SETUP",
                    "predicate": "budget",
                    "operator": "LTE",
                    "value": {"kind": "QUANTITY", "amount": 25_000_000, "unit": "VND"},
                    "start": 0,
                    "end": len(message),
                }
            ]
        }
    )
    budget = Budget()
    result = await NativeRequirementExtractor(provider, budget).extract(
        message, CanonicalRequirementState()
    )
    assert isinstance(result, Draft)
    assert result.items[0].value.amount == 25_000_000
    assert provider.model.schema is Draft
    assert budget.reservations[0][0] == "MODEL"
    assert message in provider.model.messages[-1][1]


@pytest.mark.asyncio
async def test_extraction_cannot_dispatch_when_budget_denies():
    provider = Provider({"items": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeRequirementExtractor(provider, Budget(denied=True)).extract(
            "Build PC", CanonicalRequirementState()
        )
    assert provider.model.messages == []


@pytest.mark.asyncio
async def test_invalid_native_output_is_not_silently_converted_to_empty_patch():
    provider = Provider(
        {
            "items": [
                {
                    "subject": "GPU",
                    "predicate": "identity",
                    "value": {"kind": "REFERENCE", "variant_id": "invented"},
                    "start": 0,
                    "end": 4,
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        await NativeRequirementExtractor(provider, Budget()).extract(
            "RTX GPU", CanonicalRequirementState()
        )
