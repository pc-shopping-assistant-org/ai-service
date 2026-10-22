import pytest
from pydantic import ValidationError

from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    RequirementDraft,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
    build_requirement_extraction_request,
)


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
async def test_native_extraction_uses_requirement_only_contract_and_durable_boundary():
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
    assert result.schema_version == 2
    assert result.items[0].value.amount == 25_000_000
    assert provider.model.schema is RequirementDraft
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


@pytest.mark.asyncio
async def test_requirement_extraction_schema_does_not_register_tasks():
    provider = Provider({"items": []})

    await NativeRequirementExtractor(provider, Budget()).extract(
        "Build PC 25 triệu", CanonicalRequirementState()
    )

    assert "tasks" not in provider.model.schema.model_fields


def test_requirement_extraction_prompt_does_not_ask_for_task_registration():
    prompt, _ = build_requirement_extraction_request(
        "Build PC 25 triệu", CanonicalRequirementState()
    )
    prompt = prompt.casefold()
    assert "extract requested tasks separately" not in prompt
    assert "build, compare, answer or explain_saved_build" not in prompt


def test_requirement_draft_rejects_task_registration_fields():
    with pytest.raises(ValidationError, match="tasks"):
        RequirementDraft.model_validate({"items": [], "tasks": []})
