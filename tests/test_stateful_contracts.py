from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ConversationHeadV1,
    ExecutionStopEvidenceV1,
    RunStatus,
    TurnPatchV1,
    TurnRequestV1,
    parse_turn_patch,
)


def test_sparse_patch_preserves_absent_and_clear_in_serialization():
    patch = TurnPatchV1.model_validate(
        {
            "schema_version": 1,
            "constraints": {
                "target_budget_vnd": {"op": "SET", "value": 25_000_000},
                "preferred_gpu_brand": {"op": "CLEAR"},
            },
        }
    )
    assert patch.to_payload() == {
        "schema_version": 1,
        "constraints": {
            "target_budget_vnd": {"op": "SET", "value": 25_000_000},
            "preferred_gpu_brand": {"op": "CLEAR"},
        },
    }
    assert TurnPatchV1(schema_version=1).to_payload() == {"schema_version": 1}


@pytest.mark.parametrize(
    "constraints",
    [
        None,
        {"target_budget_vnd": None},
        {"target_budget_vnd": {"op": "SET"}},
        {"target_budget_vnd": {"op": "SET", "value": None}},
        {"target_budget_vnd": {"op": "SET", "value": True}},
        {"target_budget_vnd": {"op": "SET", "value": "25000000"}},
        {"target_budget_vnd": {"op": "SET", "value": -1}},
        {"target_budget_vnd": {"op": "CLEAR", "value": 1}},
        {"target_fps": {"op": "SET", "value": 29}},
        {"source": {"op": "SET", "value": "SYSTEM"}},
        {"use_case": {"op": "SET", "value": "invented"}},
    ],
)
def test_malformed_patch_is_rejected(constraints):
    with pytest.raises(ValidationError):
        TurnPatchV1.model_validate({"schema_version": 1, "constraints": constraints})


def test_owned_and_pinned_are_distinct_refs_not_prices_or_specs():
    variant = str(uuid4())
    patch = TurnPatchV1.model_validate(
        {
            "schema_version": 1,
            "owned_parts": {"PSU": {"op": "SET", "value": {"variant_id": variant}}},
            "pinned_parts": {"GPU": {"op": "SET", "value": {"variant_id": variant}}},
        }
    )
    assert patch.owned_parts["PSU"].value.exclude_from_budget is True
    for extra in [{"price": 1}, {"exclude_from_budget": True}, {"specs": {}}]:
        with pytest.raises(ValidationError):
            TurnPatchV1.model_validate(
                {
                    "schema_version": 1,
                    "pinned_parts": {
                        "GPU": {"op": "SET", "value": {"variant_id": variant, **extra}}
                    },
                }
            )


def test_duplicate_json_keys_are_rejected_before_pydantic():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_turn_patch('{"schema_version":1,"constraints":{},"constraints":{}}')


def test_turn_does_not_accept_owner_checkpoint_or_bool_revision():
    body = {
        "conversation_id": str(uuid4()),
        "request_id": str(uuid4()),
        "expected_revision": 0,
        "message": "hello",
    }
    assert TurnRequestV1.model_validate(body).expected_revision == 0
    for extra in [
        {"account_id": str(uuid4())},
        {"checkpoint_id": "latest"},
        {"expected_revision": True},
        {"message": "   "},
    ]:
        with pytest.raises(ValidationError):
            TurnRequestV1.model_validate({**body, **extra})


def test_accepted_head_must_match_conversation_thread_and_revision():
    conversation = uuid4()
    head = {
        "conversation_id": conversation,
        "revision": 1,
        "accepted_checkpoint_ref": {"thread_id": conversation, "checkpoint_id": "cp-a"},
    }
    assert ConversationHeadV1.model_validate(head).thread_id == str(conversation)
    for ref in [
        None,
        {"thread_id": uuid4(), "checkpoint_id": "cp-a"},
        {"thread_id": conversation, "checkpoint_id": "cp-a", "checkpoint_ns": "debug"},
    ]:
        with pytest.raises(ValidationError):
            ConversationHeadV1.model_validate({**head, "accepted_checkpoint_ref": ref})


def test_native_checkpoint_identity_is_not_normalized():
    ref = CheckpointRefV1(thread_id=uuid4(), checkpoint_id="opaque:checkpoint ")
    assert ref.checkpoint_id == "opaque:checkpoint "


def test_stop_evidence_requires_explicit_drained_writes_and_execution_identity():
    body = {
        "kind": "EXECUTOR_STOP_ACK",
        "execution_id": uuid4(),
        "worker_instance_id": uuid4(),
        "generation": 1,
        "stopped_at": datetime.now(UTC),
        "checkpoint_writes_drained": True,
        "proof_ref": "internal:task-joined",
    }
    assert (
        ExecutionStopEvidenceV1.model_validate(body).checkpoint_writes_drained is True
    )
    for extra in [
        {"checkpoint_writes_drained": False},
        {"checkpoint_writes_drained": 1},
        {"generation": 0},
        {"proof_ref": ""},
        {"stopped_at": datetime.now(UTC).replace(tzinfo=None)},
    ]:
        with pytest.raises(ValidationError):
            ExecutionStopEvidenceV1.model_validate({**body, **extra})
    del body["checkpoint_writes_drained"]
    with pytest.raises(ValidationError):
        ExecutionStopEvidenceV1.model_validate(body)


def test_failure_statuses_remain_distinct():
    assert RunStatus.FAILED_RETRYABLE != RunStatus.FAILED_TERMINAL
    with pytest.raises(ValueError):
        RunStatus("FAILED")


@pytest.mark.parametrize(
    "body",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"schema_version": 1, "owned_parts": None},
        {"schema_version": 1, "owned_parts": {"MONITOR": {"op": "CLEAR"}}},
        {
            "schema_version": 1,
            "owned_parts": {
                "GPU": {
                    "op": "SET",
                    "value": {
                        "variant_id": str(uuid4()),
                        "exclude_from_budget": "true",
                    },
                }
            },
        },
        {"schema_version": 1, "phase": "READY_TO_PUBLISH"},
    ],
)
def test_patch_version_slots_bool_and_server_fields_are_strict(body):
    with pytest.raises(ValidationError):
        TurnPatchV1.model_validate(body)


def test_json_schema_forbids_extra_fields_and_has_typed_operations():
    schema = TurnPatchV1.model_json_schema()
    assert schema["additionalProperties"] is False
    constraints = schema["$defs"]["ConstraintsPatchV1"]
    assert constraints["additionalProperties"] is False
    operation_ref = constraints["properties"]["target_budget_vnd"]["anyOf"][0]["$ref"]
    operation = schema["$defs"][operation_ref.rsplit("/", 1)[-1]]
    assert operation["discriminator"]["propertyName"] == "op"
    assert set(operation["discriminator"]["mapping"]) == {"SET", "CLEAR"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"constraints":{"use_case":{"op":"CLEAR","op":"SET"}}}',
        '{"schema_version":1,"constraints":{"target_budget_vnd":{"op":"SET","value":NaN}}}',
    ],
)
def test_noncanonical_json_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_turn_patch(raw)
