from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.public_contracts import (
    PublicBuildPartV1,
    PublicConsultationV1,
    StatefulEventV1,
    TurnResultV1,
    to_public_consultation,
)
from ai_service.schemas.response import ApiResponse, ResponseMessage


def test_public_state_is_allowlisted_and_does_not_publish_execution_internals():
    state = ConsultationStateV1()
    public = to_public_consultation(state)
    assert public.target_budget_vnd is None
    assert public.budget_scope == "BUILD_PC"
    assert set(public.model_dump()) == {
        "budget_scope",
        "target_budget_vnd",
        "accessory_budget_vnd",
        "use_case",
        "target_resolution",
        "target_fps",
        "preferred_gpu_brand",
        "preferred_cpu_brand",
        "form_factor_preference",
        "owned_parts",
        "pinned_parts",
        "accessories",
        "build_stale",
    }
    with pytest.raises(ValidationError):
        PublicConsultationV1.model_validate(
            {**public.model_dump(), "checkpoint_id": "cp-secret"}
        )


def test_owned_public_build_is_free_but_pinned_cannot_be_marked_owned():
    body = {
        "slot": "GPU",
        "variant_id": uuid4(),
        "name": "GPU",
        "spending_vnd": 1,
        "is_owned": True,
        "is_pinned": False,
    }
    with pytest.raises(ValidationError):
        PublicBuildPartV1.model_validate(body)
    assert PublicBuildPartV1.model_validate({**body, "spending_vnd": 0}).is_owned
    with pytest.raises(ValidationError):
        PublicBuildPartV1.model_validate({**body, "spending_vnd": 0, "is_pinned": True})


def test_sse_delta_is_correlated_typed_and_uses_the_existing_envelope():
    event = TypeAdapter(StatefulEventV1).validate_python(
        {
            "event": "DELTA",
            "conversation_id": uuid4(),
            "run_id": uuid4(),
            "sequence": 1,
            "delta": "hello",
        }
    )
    response = ApiResponse[StatefulEventV1](
        data=event, message=ResponseMessage.AI_CHAT_STREAM_DELTA
    )
    assert set(response.model_dump()) == {"data", "message", "errors"}
    whitespace = TypeAdapter(StatefulEventV1).validate_python(
        {**event.model_dump(), "delta": " "}
    )
    assert whitespace.delta == " "  # Streaming whitespace must not be stripped.
    with pytest.raises(ValidationError):
        TypeAdapter(StatefulEventV1).validate_python(
            {**event.model_dump(), "lease_generation": 2}
        )


def test_failed_or_unpublished_run_cannot_be_a_successful_turn_result():
    body = {
        "run": {
            "run_id": uuid4(),
            "conversation_id": uuid4(),
            "request_id": uuid4(),
            "status": "FAILED_RETRYABLE",
            "base_revision": 0,
        },
        "revision": 1,
        "state": to_public_consultation(ConsultationStateV1()),
        "answer": "saved",
    }
    with pytest.raises(ValidationError):
        TurnResultV1.model_validate(body)
    body["run"]["status"] = "COMPLETED"
    body["revision"] = 0
    with pytest.raises(ValidationError):
        TurnResultV1.model_validate(body)
