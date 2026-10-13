from uuid import uuid4

import pytest

from ai_service.capabilities.assistant.react.turn_contracts import (
    TurnRequestV2,
    parse_turn_request,
    turn_request_payload,
)
from ai_service.capabilities.assistant.stateful_contracts import TurnRequestV1


def payload():
    return {
        "conversation_id": uuid4(),
        "request_id": uuid4(),
        "expected_revision": 0,
        "message": "OK",
    }


def confirmation():
    return {
        "proposal_id": uuid4(),
        "proposal_hash": "a" * 64,
        "target_revision": 1,
        "base_requirements_hash": "b" * 64,
    }


def test_v2_confirmation_round_trip_keeps_complete_immutable_binding():
    request = TurnRequestV2(**payload(), confirmation=confirmation())
    encoded = turn_request_payload(request)
    restored = parse_turn_request(encoded)
    assert restored == request
    assert encoded["schema_version"] == 2
    assert restored.confirmation.proposal_hash == "a" * 64
    assert "unit_reply" not in encoded
    assert restored.execution_input()["message_id"] == str(request.request_id)


def test_legacy_payload_and_idempotency_shape_remain_unchanged():
    request = TurnRequestV1(**payload())
    encoded = turn_request_payload(request)
    assert set(encoded) == {
        "conversation_id",
        "request_id",
        "expected_revision",
        "message",
    }
    assert type(parse_turn_request(encoded)) is TurnRequestV1


@pytest.mark.parametrize("version", [1, 3, True, "2", None])
def test_explicit_unknown_or_non_integer_schema_never_falls_back_to_v1(version):
    with pytest.raises(ValueError):
        parse_turn_request(payload() | {"schema_version": version})


def test_confirmation_cannot_mix_with_task_selection():
    with pytest.raises(ValueError):
        TurnRequestV2(
            **payload(),
            confirmation=confirmation(),
            task_reply={
                "pending_message_id": uuid4(),
                "constraints_revision": 0,
                "selections": [
                    {"task_index": 0, "subject_index": 0, "product_id": uuid4()}
                ],
            },
        )


def test_model_thread_checkpoint_or_owner_fields_are_not_client_selectors():
    with pytest.raises(ValueError):
        TurnRequestV2(**payload(), checkpoint_id="foreign", owner=uuid4())
