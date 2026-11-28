"""Model interaction is text/native calls; application owns business outcomes."""

import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from test_native_react_model import Provider
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    PendingConversationContext,
    ToolCall,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.control_actions import normalize_control
from ai_service.capabilities.assistant.react.dialogue import pending_context
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.diagnostics import CompilationReason
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.providers.react_model import (
    NativeFlexibleModel,
    build_react_request,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
async def test_plain_text_becomes_response_without_fabricating_business_outcome(
    provider_name,
):
    text = "Chào bạn! Bạn cần tư vấn cấu hình hay chọn linh kiện?"
    provider = Provider(AIMessage(content=text), name=provider_name)
    decision = await NativeFlexibleModel(provider).decide(
        "Chào bạn", [], (ToolName.SEARCH_CATALOG,), None, context()
    )
    assert decision is not None
    assert decision.action.kind == "RESPOND"
    assert decision.action.text == text
    assert not hasattr(decision.action, "outcome")
    assert not hasattr(decision.action, "task_outcomes")


def test_model_request_exposes_small_native_controls_without_decision_envelope():
    from test_saved_build_explanation import saved_context

    trusted = saved_context()
    trusted.completion_requirements = [CompletionRequirement(kind="ANSWER")]
    messages, schemas = build_react_request(
        "GPU?", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    system = messages[0].content
    assert "FlexibleDecision" not in system
    assert '"discriminator"' not in system
    by_name = {schema["name"]: schema for schema in schemas}
    assert {
        "search_catalog",
        "propose_requirements",
        "submit_result",
    } <= by_name.keys()
    assert "ask_clarification" not in by_name
    for name in ("propose_requirements", "submit_result"):
        schema = json.dumps(by_name[name]["parameters"])
        assert "FlexibleDecision" not in schema
        assert '"outcome"' not in schema
        assert '"status"' not in schema
        assert "schema_version" not in by_name[name]["parameters"].get("properties", {})


def test_model_gets_pending_build_fields_when_user_answers_a_clarification():
    trusted = context()
    trusted.completion_requirements = [
        CompletionRequirement(kind="BUILD", constraints_revision=0)
    ]
    trusted.pending = pending_context(
        FlexiblePublishedState(
            conversation_id=trusted.conversation_id,
            task_message_id=uuid4(),
            task_message="Build PC",
            status="WAITING_INPUT",
            waiting_message="Ngân sách tối đa của bạn là bao nhiêu?",
            waiting_tasks=[
                TaskOutcome(
                    task=trusted.completion_requirements[0], status="UNRESOLVED"
                )
            ],
            clarification={
                "question": "Ngân sách tối đa của bạn là bao nhiêu?",
                "missing_fields": [CompilationReason.BUDGET_REQUIRED.value],
            },
        )
    )

    messages, _ = build_react_request("30 triệu", [], (), None, trusted)
    instruction, data = messages[0].content.split(" Application context (data): ", 1)
    assert json.loads(data)["pending"]["missing_fields"] == ["BUDGET_REQUIRED"]
    assert "call propose_requirements before replying" in instruction


@pytest.mark.parametrize(
    ("message", "expected_operator"), [("30 triệu", "LTE"), ("Đúng 30 triệu", "EQ")]
)
def test_bare_budget_reply_uses_pending_spending_cap(message, expected_operator):
    trusted = context()
    trusted.pending = PendingConversationContext(
        question="Ngân sách tối đa của bạn là bao nhiêu?",
        missing_fields=[CompilationReason.BUDGET_REQUIRED],
    )
    call = ToolCall(
        call_id="budget",
        name="propose_requirements",
        arguments={
            "items": [
                {
                    "subject": "SETUP",
                    "predicate": "budget",
                    "value": 30_000_000,
                    "unit": "VND",
                    "operator": "EQ",
                    "quote": message,
                }
            ]
        },
    )
    decision = normalize_control(call, [], message, trusted)
    assert decision.action.draft.items[0].operator == expected_operator


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
async def test_native_control_call_normalizes_to_internal_action(provider_name):
    trusted = context()
    control = "propose_requirements"
    arguments = {
        "source_text": "RAM 32GB",
        "items": [
            {
                "subject": "RAM",
                "predicate": "capacity",
                "quote": "RAM 32GB",
                "value": 32,
                "unit": "GB",
            }
        ],
    }
    before = trusted.model_dump(mode="json")
    provider = Provider(
        AIMessage(
            content="Accompanying text is not a separate response",
            tool_calls=[{"id": "control", "name": control, "args": arguments}],
        ),
        name=provider_name,
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Tư vấn PC", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    assert decision is not None
    assert decision.action.kind == control.upper()
    assert decision.action.draft.items[0].value.amount == 32
    assert decision.action.source_text == "RAM 32GB"
    assert trusted.model_dump(mode="json") == before
    assert "search_catalog" in {tool["name"] for tool in provider.model.tools}


@pytest.mark.asyncio
async def test_submit_result_selects_canonical_evidence_without_model_outcome():
    trusted, catalog = context(), Catalog()
    trusted.completion_requirements = [CompletionRequirement(kind="ANSWER")]
    call = ToolCall(
        call_id="catalog-read", name="search_catalog", arguments={"query": "GPU"}
    )
    result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        call, trusted
    )
    record = ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "submit",
                    "name": "submit_result",
                    "args": {
                        "results": [
                            {
                                "task_index": 0,
                                "text": "Sản phẩm hiện có trong catalog.",
                                "refs": result.evidence_refs,
                            }
                        ]
                    },
                }
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU?", [record], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    assert decision is not None
    assert decision.action.kind == "FINAL_ANSWER"
    assert decision.action.outcome == "ANSWERED"
    assert decision.action.answer.evidence_refs == result.evidence_refs
    assert catalog.calls == 1


@pytest.mark.asyncio
async def test_selecting_build_ref_resolves_its_canonical_refs_before_explanation(
    realistic_catalog,
):
    from test_requirement_pipeline import rows, state

    from ai_service.capabilities.assistant.react.actions import FlexibleDecision
    from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
    from ai_service.capabilities.assistant.react.submissions import (
        ResultSelection,
        SubmitResults,
        derive_final_action,
        evidence_requirements,
    )

    trusted = context(state(), rows(realistic_catalog))
    service = PCBuildApplicationService()
    call = ToolCall(call_id="build", name="build_pc", arguments={})
    result = await ToolDispatcher(Catalog(), service).execute(call, trusted)
    record = ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)
    selected = SubmitResults(
        results=[
            ResultSelection(
                text="Untrusted price 1 VND",
                build_ref=result.result.build_ref,
                explanation="Lựa chọn bám theo ngân sách.",
            )
        ]
    )
    action = derive_final_action(selected, [record])
    assert action.answer.evidence_refs == result.evidence_refs
    validated = validate_final_answer(
        action,
        [result],
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        service,
    )
    assert validated.outcome == "BUILT"
    assert "Untrusted price" not in validated.text

    fallback = FlexibleDecision(
        action={"kind": "RESPOND", "text": "Đây là cấu hình 1 VND"}
    )
    obligations = evidence_requirements(fallback, [record], trusted)
    assert [task.kind for task in obligations] == ["BUILD"]


@pytest.mark.asyncio
async def test_multiple_control_calls_reject_entire_provider_response():
    call = {
        "name": "propose_requirements",
        "args": {
            "items": [
                {
                    "subject": "SETUP",
                    "predicate": "budget",
                    "operator": "LTE",
                    "value": 20_000_000,
                    "unit": "VND",
                    "quote": "20 triệu",
                }
            ]
        },
    }
    provider = Provider(
        AIMessage(
            content="Do not publish this accompanying response",
            tool_calls=[call | {"id": "one"}, call | {"id": "two"}],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Build PC 20 triệu", [], (ToolName.SEARCH_CATALOG,), None, context()
    )
    assert decision is None


@pytest.mark.asyncio
async def test_grounded_side_answer_does_not_complete_the_pending_build():
    from ai_service.capabilities.assistant.react.dialogue import validate_response

    trusted, catalog = context(), Catalog()
    task = CompletionRequirement(kind="BUILD", constraints_revision=0)
    trusted.completion_requirements = [task]
    call = ToolCall(call_id="price", name="search_catalog", arguments={"query": "GPU"})
    result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        call, trusted
    )
    records = [ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)]
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "side-result",
                    "name": "submit_result",
                    "args": {
                        "results": [
                            {
                                "task_index": None,
                                "text": "Giá tra từ catalog.",
                                "refs": result.evidence_refs,
                            }
                        ]
                    },
                }
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Giá con đó hiện tại?",
        records,
        (ToolName.SEARCH_CATALOG,),
        None,
        trusted,
    )
    assert decision.action.kind == "RESPOND"
    response = validate_response(
        decision.action, trusted, [task], records, continuing_pending=True
    )
    assert response.text == "Giá tra từ catalog."
    assert response.evidence_refs == result.evidence_refs
    assert not hasattr(response, "outcome")
    with pytest.raises(ValueError, match="business result"):
        validate_response(decision.action, trusted, [task], records)


def test_submit_result_tool_schema_requires_explanation_first():
    from ai_service.capabilities.assistant.react.model_contracts import (
        EvidenceSubmissionInput,
    )
    from ai_service.infrastructure.providers.tool_schema import model_tool_schema

    schema = model_tool_schema(EvidenceSubmissionInput)
    item_schema = schema["properties"]["results"]["items"]
    assert next(iter(item_schema["properties"])) == "explanation"
    assert "text" not in item_schema["properties"]
    assert "explanation" in item_schema["required"]


@pytest.mark.parametrize(
    ("message", "expected_operator"),
    [("25 triệu", "LTE"), ("Đúng 25 triệu", "EQ")],
)
def test_bare_budget_reply_without_pending_context_normalizes_eq_to_lte(
    message, expected_operator
):
    trusted = context()
    assert trusted.pending is None
    call = ToolCall(
        call_id="budget",
        name="propose_requirements",
        arguments={
            "items": [
                {
                    "subject": "SETUP",
                    "predicate": "budget",
                    "value": 25_000_000,
                    "unit": "VND",
                    "operator": "EQ",
                    "quote": message,
                }
            ]
        },
    )
    decision = normalize_control(call, [], message, trusted)
    assert decision.action.draft.items[0].operator == expected_operator


def test_unitless_quantity_quote_downgrades_guessed_unit_and_formats_vietnamese():
    from ai_service.capabilities.assistant.react.waiting_publication import (
        _format_pending_resolution_message,
    )
    from ai_service.capabilities.assistant.requirements.resolver import resolve_draft
    from ai_service.capabilities.assistant.requirements.units import (
        UnresolvedQuantityValue,
    )

    message = "Build PC 25 triệu chơi game 1080p, RAM 32"
    call = ToolCall(
        call_id="ram_no_unit",
        name="propose_requirements",
        arguments={
            "items": [
                {
                    "subject": "RAM",
                    "predicate": "capacity",
                    "value": 32,
                    "unit": "GB",
                    "quote": "RAM 32",
                }
            ]
        },
    )
    decision = normalize_control(call, [], message, context())
    assert isinstance(decision.action.draft.items[0].value, UnresolvedQuantityValue)
    assert decision.action.draft.items[0].value.amount == 32

    resolution = resolve_draft(
        decision.action.draft,
        message_id=uuid4(),
        message=message,
        state=context().requirements,
        catalog=[],
    )
    assert resolution.patch is None
    assert [item.reason for item in resolution.pending] == ["QUANTITY_UNIT_REQUIRED"]
    rendered = _format_pending_resolution_message(
        ["QUANTITY_UNIT_REQUIRED", "QUANTITY_EVIDENCE_MISMATCH"]
    )
    assert "QUANTITY_" not in rendered
    assert "đơn vị" in rendered


@pytest.mark.asyncio
async def test_submit_result_tolerates_empty_explanation_refs_tool_name_and_raw_build_uuid(
    realistic_catalog,
):
    from test_requirement_pipeline import rows, state

    trusted = context(state(), rows(realistic_catalog))
    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    build_call = ToolCall(call_id="build", name="build_pc", arguments={})
    build_result = await dispatcher.execute(build_call, trusted)
    records = [
        ToolRecord(call=build_call, result=build_result, origin_run_id=trusted.run_id)
    ]
    build_ref = str(build_result.result.build_ref)

    messages, _ = build_react_request(
        "Build PC 25 triệu chơi game 1080p",
        records,
        (ToolName.SEARCH_CATALOG,),
        None,
        trusted,
    )
    assert f'"evidence_ref": "build:{build_ref}"' in messages[0].content

    for args in (
        {
            "results": [
                {
                    "build_ref": build_ref,
                    "explanation": "Cấu hình phù hợp gaming 1080p.",
                    "explanation_refs": [],
                }
            ]
        },
        {
            "results": [
                {
                    "calls": ["build_pc"],
                    "explanation": "Cấu hình phù hợp gaming 1080p.",
                }
            ]
        },
        {
            "results": [
                {
                    "refs": [build_ref],
                    "explanation": "Cấu hình phù hợp gaming 1080p.",
                }
            ]
        },
    ):
        call = ToolCall(call_id="submit", name="submit_result", arguments=args)
        decision = normalize_control(call, records, "Build PC", trusted)
        assert decision.action.kind == "FINAL_ANSWER"
        assert decision.action.outcome == "BUILT"
        assert decision.action.answer.build_ref == build_result.result.build_ref
        assert decision.action.answer.evidence_refs == [f"build:{build_ref}"]
        assert decision.action.answer.explanation is not None
        assert decision.action.answer.explanation.evidence_refs == [
            f"build:{build_ref}"
        ]


@pytest.mark.asyncio
async def test_unresolved_tasks_plain_text_transitions_to_waiting_input():
    from test_flexible_publication import base_snapshot
    from test_react_graph import Budget

    from ai_service.capabilities.assistant.react.actions import (
        FlexibleDecision,
        RespondAction,
    )
    from ai_service.capabilities.assistant.react.snapshot_publication import (
        FlexiblePublicationInput,
    )
    from ai_service.capabilities.assistant.requirements.tasks import (
        TaskDraft,
        TaskDraftBatch,
        TaskEvidence,
    )
    from ai_service.infrastructure.graph.flexible import build_flexible_graph

    class AmbiguousCatalog(Catalog):
        async def resolve(self, mention: str):
            second = self.product.model_copy(
                update={
                    "product_id": uuid4(),
                    "name": "MSI RTX 4060 Ventus",
                    "evidence_ref": "catalog:product:second",
                }
            )
            return [self.product, second]

    class Model:
        async def decide(self, message, transcript, allowed, repair, ctx=None):
            return FlexibleDecision(
                action=RespondAction(text="Bạn muốn chọn mẫu RTX 4060 nào?"),
                registration=TaskDraftBatch(
                    tasks=[
                        TaskDraft(
                            kind="COMPARE",
                            start=0,
                            end=len(message),
                            quote=message,
                            subjects=[
                                TaskEvidence(start=8, end=16, quote="RTX 4060"),
                                TaskEvidence(start=20, end=28, quote="RTX 4070"),
                            ],
                        )
                    ]
                ),
            )

    trusted, products = context(), AmbiguousCatalog()
    message = "So sánh RTX 4060 và RTX 4070"
    flow = build_flexible_graph(
        Model(),
        ToolDispatcher(products, PCBuildApplicationService()),
        trusted,
        [],
        Budget(),
        conversation_revision=0,
        task_catalog=products,
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(trusted),
            task_message_id=uuid4(),
            task_message=message,
            request_message=message,
        ),
    )
    terminal = await flow.ainvoke({"message": message})
    assert terminal["phase"] == "WAITING_INPUT"
    assert terminal["clarification"]["missing_fields"] == ["AMBIGUOUS"]
    assert terminal["accepted_snapshot"]["flexible"]["status"] == "WAITING_INPUT"
    assert (
        terminal["accepted_snapshot"]["flexible"]["pending_task_resolution"]
        is not None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "expected_method"),
    [("gemini", "function_calling"), ("openai", "json_schema")],
)
async def test_structured_extractors_use_provider_compatible_output_method(
    provider_name, expected_method
):
    from ai_service.capabilities.assistant.public_contracts import (
        ClarificationItemV1,
        PublicClarificationV1,
    )
    from ai_service.capabilities.assistant.source_quotes import (
        QuotedTaskBatch,
        QuotedUnitChoice,
        QuotedUnitSelection,
    )
    from ai_service.infrastructure.providers.requirement_extractor import (
        NativeRequirementExtractor,
    )
    from ai_service.infrastructure.providers.task_extractor import NativeTaskExtractor

    calls: list[tuple[type, str | None]] = []

    class PassThroughAttempts:
        async def invoke(self, kind, operation, key, dispatch):
            return await dispatch()

    class FakeStructuredModel:
        def __init__(self, response):
            self.response = response

        async def ainvoke(self, messages):
            return self.response

    class FakeChatModel:
        def __init__(self, responses):
            self.responses = iter(responses)

        def with_structured_output(self, schema, **kwargs):
            calls.append((schema, kwargs.get("method")))
            return FakeStructuredModel(next(self.responses))

    class FakeProvider:
        def __init__(self, responses):
            self.name = provider_name
            self._chat = FakeChatModel(responses)

        def create_model(self):
            return self._chat

    provider = FakeProvider(
        [
            QuotedUnitChoice(
                selections=[
                    QuotedUnitSelection(item_index=0, unit="GB", quote="GB"),
                ]
            ),
            QuotedTaskBatch(tasks=[]),
        ]
    )
    options = PublicClarificationV1(
        pending_message_id=uuid4(),
        constraints_revision=0,
        items=[
            ClarificationItemV1(
                item_index=0,
                reason="QUANTITY_UNIT_REQUIRED",
                quantity_amount=32,
                unit_options=["GB", "TB"],
            )
        ],
    )
    req_extractor = NativeRequirementExtractor(provider, PassThroughAttempts())
    unit_draft = await req_extractor.choose_unit("32 GB", options)
    assert unit_draft.selections[0].unit == "GB"

    task_extractor = NativeTaskExtractor(provider, PassThroughAttempts())
    task_batch = await task_extractor.extract_tasks("Chào bạn")
    assert task_batch.tasks == []

    assert calls == [
        (QuotedUnitChoice, expected_method),
        (QuotedTaskBatch, expected_method),
    ]


@pytest.mark.asyncio
async def test_proposal_confirmation_without_waiting_tasks_auto_clarifies_core_build():
    from test_flexible_publication import base_snapshot
    from test_react_graph import Budget

    from ai_service.capabilities.assistant.react.snapshot_publication import (
        FlexiblePublicationInput,
        revalidate_flexible_publication,
    )
    from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
    from ai_service.capabilities.assistant.requirements.contracts import (
        Evidence,
        QuantityValue,
        Requirement,
        RequirementPatch,
    )
    from ai_service.capabilities.assistant.requirements.proposals import (
        ProposalConfirmation,
        issue_proposal,
    )
    from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
    from ai_service.infrastructure.graph.flexible import build_flexible_graph

    trusted = context()
    proposal_msg_id = uuid4()
    proposal_text = "Build PC 25 triệu"
    patch = RequirementPatch(
        set_requirements=[
            Requirement(
                id=uuid4(),
                subject="SETUP",
                predicate="budget",
                operator="LTE",
                value=QuantityValue(amount=25_000_000, unit="VND"),
                strength="HARD",
                evidence=Evidence(
                    message_id=proposal_msg_id,
                    start=9,
                    end=17,
                    quote="25 triệu",
                ),
            )
        ]
    )
    pending = issue_proposal(
        trusted.requirements,
        patch,
        {proposal_msg_id: proposal_text},
        owner=trusted.account_id,
        conversation_id=trusted.conversation_id,
        target_revision=trusted.requirements.revision,
    )
    accepted = base_snapshot(trusted).model_copy(
        update={
            "flexible": FlexiblePublishedState(
                conversation_id=trusted.conversation_id,
                task_message_id=proposal_msg_id,
                task_message=proposal_text,
                status="WAITING_INPUT",
                waiting_message="Xác nhận ngân sách 25 triệu?",
                waiting_tasks=[],
                pending_proposal=pending,
            )
        }
    )
    confirmation = ProposalConfirmation(
        proposal_id=pending.proposal_id,
        proposal_hash=pending.proposal_hash,
        target_revision=pending.target_revision,
        base_requirements_hash=pending.base_requirements_hash,
    )

    class FailIfCalledModel:
        async def decide(self, *args, **kwargs):
            raise AssertionError("Model must not be called on proposal confirmation")

    catalog = Catalog()
    service = PCBuildApplicationService()
    flow = build_flexible_graph(
        FailIfCalledModel(),
        ToolDispatcher(catalog, service),
        trusted,
        [],
        Budget(),
        conversation_revision=0,
        task_catalog=catalog,
        publication=FlexiblePublicationInput(
            accepted=accepted,
            task_message_id=proposal_msg_id,
            task_message=proposal_text,
            request_message="Xác nhận đề xuất",
            confirmation=confirmation,
        ),
    )
    terminal = await flow.ainvoke({"message": "Xác nhận đề xuất"})
    assert terminal["phase"] == "WAITING_INPUT"
    assert terminal["clarification"]["missing_fields"] == ["USE_CASE_REQUIRED"]
    assert terminal["accepted_snapshot"]["flexible"]["status"] == "WAITING_INPUT"
    assert terminal["accepted_snapshot"]["flexible"]["pending_proposal"] is None
    assert terminal["accepted_snapshot"]["requirement_state"]["revision"] == 1

    revalidated = revalidate_flexible_publication(
        terminal,
        accepted,
        TurnRequestV2(
            conversation_id=trusted.conversation_id,
            request_id=uuid4(),
            expected_revision=0,
            message="Xác nhận đề xuất",
            confirmation=confirmation,
        ),
        owner=trusted.account_id,
        run_id=trusted.run_id,
        base_ref=CheckpointRefV1(
            thread_id=trusted.conversation_id, checkpoint_id="cp-0"
        ),
        service=service,
    )
    assert revalidated.flexible is not None
    assert revalidated.flexible.clarification is not None
    assert revalidated.flexible.clarification.missing_fields == ["USE_CASE_REQUIRED"]
    assert revalidated.flexible.pending_proposal is None


@pytest.mark.asyncio
async def test_infeasible_build_submission_narrows_mixed_search_catalog_refs(
    realistic_catalog,
):
    from test_requirement_pipeline import requirement, rows

    from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
    from ai_service.capabilities.assistant.requirements.contracts import (
        CanonicalRequirementState,
        QuantityValue,
        TextValue,
    )

    budget = requirement(
        "SETUP",
        "budget",
        QuantityValue(amount=500_000, unit="VND"),
        operator="LTE",
    )
    goal = requirement("WORKLOAD", "use_case", TextValue(text="GAMING_1080P"))
    req_state = CanonicalRequirementState(
        requirements={budget.id: budget},
        goals={goal.id: goal},
    )
    trusted = context(req_state, rows(realistic_catalog))
    service = PCBuildApplicationService()
    dispatcher = ToolDispatcher(Catalog(), service)

    build_call = ToolCall(call_id="build", name="build_pc", arguments={})
    build_result = await dispatcher.execute(build_call, trusted)
    assert build_result.result.search_evidence is not None
    assert build_result.result.search_evidence.outcome == "INFEASIBLE"

    search_call = ToolCall(
        call_id="search", name="search_catalog", arguments={"query": "GPU"}
    )
    search_result = await dispatcher.execute(search_call, trusted)
    records = [
        ToolRecord(call=build_call, result=build_result, origin_run_id=trusted.run_id),
        ToolRecord(
            call=search_call, result=search_result, origin_run_id=trusted.run_id
        ),
    ]

    submit_call = ToolCall(
        call_id="submit",
        name="submit_result",
        arguments={
            "results": [
                {
                    "build_ref": str(build_result.result.build_ref),
                    "refs": [
                        *build_result.evidence_refs,
                        *search_result.evidence_refs,
                    ],
                    "explanation": "Ngân sách 500k không đủ để build PC gaming.",
                }
            ]
        },
    )
    decision = normalize_control(submit_call, records, "Build PC 500k gaming", trusted)
    assert decision.action.kind == "FINAL_ANSWER"
    assert decision.action.outcome == "INFEASIBLE"
    assert decision.action.answer.evidence_refs == build_result.evidence_refs
    assert decision.action.answer.explanation is not None
    assert (
        decision.action.answer.explanation.evidence_refs == build_result.evidence_refs
    )

    validated = validate_final_answer(
        decision.action,
        [build_result, search_result],
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        service,
    )
    assert validated.outcome == "INFEASIBLE"
