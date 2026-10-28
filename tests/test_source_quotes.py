"""Model outputs quote source text; only code assigns character offsets."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.capabilities.assistant.react.control_actions import (
    QuotedProposeRequirementsAction,
)
from ai_service.capabilities.assistant.source_quotes import (
    QuotedChoice,
    QuotedPendingAmendment,
    QuotedRequirementDraft,
    QuotedTaskAmendment,
    QuotedTaskBatch,
    QuotedTaskChoice,
    QuotedUnitChoice,
    quote_span,
)


def test_unicode_quote_binding_uses_python_character_positions():
    message = "Tư vấn RAM 32GB nhé"
    assert quote_span(message, "RAM 32GB") == {"start": 7, "end": 15}


@pytest.mark.parametrize("message, quote", [("RAM RAM", "RAM"), ("Build PC", "GPU")])
def test_missing_or_repeated_quote_is_not_silently_bound(message, quote):
    with pytest.raises(ValueError, match="Ambiguous|not found"):
        quote_span(message, quote)


def test_requirement_draft_binds_quote_and_rejects_model_offsets():
    source = "Tư vấn RAM 32GB nhé"
    item = {
        "subject": "RAM",
        "predicate": "capacity",
        "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
        "quote": "RAM 32GB",
    }
    bound = QuotedRequirementDraft.model_validate({"items": [item]}).bind(source)
    assert (bound.items[0].start, bound.items[0].end) == (7, 15)
    with pytest.raises(ValidationError, match="Extra inputs"):
        QuotedRequirementDraft.model_validate(
            {"items": [{**item, "start": 7, "end": 15}]}
        )


def test_task_subject_binds_only_inside_task_quote():
    source = "RTX 4060; so RTX 4060 với RTX 4070"
    batch = QuotedTaskBatch.model_validate(
        {
            "tasks": [
                {
                    "kind": "COMPARE",
                    "quote": "so RTX 4060 với RTX 4070",
                    "subjects": [{"quote": "RTX 4060"}, {"quote": "RTX 4070"}],
                }
            ]
        }
    ).bind(source)
    assert batch.tasks[0].subjects[0].start == source.index("RTX 4060", 10)


def test_amendment_binds_both_replacement_and_discard_quotes():
    source = "Đổi RAM 64GB, bỏ SSD"
    bound = QuotedPendingAmendment.model_validate(
        {
            "replacements": [
                {
                    "item_index": 0,
                    "item": {
                        "subject": "RAM",
                        "predicate": "capacity",
                        "quote": "RAM 64GB",
                        "value": {"kind": "QUANTITY", "amount": 64, "unit": "GB"},
                    },
                }
            ],
            "discards": [{"item_index": 1, "quote": "bỏ SSD"}],
        }
    ).bind(source)
    assert (
        source[bound.replacements[0].item.start : bound.replacements[0].item.end]
        == "RAM 64GB"
    )
    assert source[bound.discards[0].start : bound.discards[0].end] == "bỏ SSD"


def test_choice_and_unit_replies_bind_exact_quotes():
    message = "Tôi chọn bản thứ hai, đơn vị GB"
    variant_id = uuid4()
    choice = QuotedChoice.model_validate(
        {
            "selections": [
                {"item_index": 1, "variant_id": variant_id, "quote": "bản thứ hai"}
            ]
        }
    ).bind(message)
    unit = QuotedUnitChoice.model_validate(
        {"selections": [{"item_index": 2, "unit": "GB", "quote": "GB"}]}
    ).bind(message)
    assert (
        message[choice.selections[0].start : choice.selections[0].end] == "bản thứ hai"
    )
    assert message[unit.selections[0].start : unit.selections[0].end] == "GB"


def test_task_choice_and_edit_bind_exact_quotes():
    message = "Chọn RTX 4070"
    product_id = uuid4()
    choice = QuotedTaskChoice.model_validate(
        {
            "selections": [
                {
                    "task_index": 0,
                    "subject_index": 1,
                    "product_id": product_id,
                    "evidence": {"quote": "RTX 4070"},
                }
            ]
        }
    ).bind(message)
    edit = QuotedTaskAmendment.model_validate(
        {
            "edits": [
                {"task_index": 0, "subject_index": 1, "evidence": {"quote": "RTX 4070"}}
            ]
        }
    ).bind(message)
    assert choice.selections[0].evidence.start == 5
    assert edit.edits[0].evidence.end == len(message)


@pytest.mark.parametrize(
    "schema",
    [
        QuotedTaskBatch,
        QuotedRequirementDraft,
        QuotedPendingAmendment,
        QuotedChoice,
        QuotedUnitChoice,
        QuotedTaskChoice,
        QuotedTaskAmendment,
        QuotedProposeRequirementsAction,
    ],
)
def test_model_facing_schemas_never_request_offsets(schema):
    components = [schema.model_json_schema()]
    while components:
        component = components.pop()
        if isinstance(component, dict):
            assert "start" not in component.get("properties", {})
            assert "end" not in component.get("properties", {})
            components.extend(component.values())
        elif isinstance(component, list):
            components.extend(component)
