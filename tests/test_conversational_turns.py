"""A turn can answer the user without completing or replacing business work."""

from uuid import uuid4

import pytest
from test_flexible_contracts import pending_proposal
from test_flexible_publication import base_snapshot
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.actions import (
    FlexibleDecision,
    RespondAction,
)
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
)
from ai_service.capabilities.assistant.react.dialogue import validate_response
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    validate_answer,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


def test_task_extraction_can_explicitly_return_no_business_tasks():
    draft = TaskDraftBatch.model_validate({"tasks": []})
    draft.validate_source("Chào bạn")
    assert draft.tasks == []
    assert TaskDraftBatch.model_validate_json(draft.model_dump_json()) == draft


def test_conversational_answer_does_not_require_business_evidence():
    text = "Chào bạn! Bạn muốn tìm linh kiện hay tư vấn cấu hình PC?"
    answer = validate_response(RespondAction(text=text), context(), [], [])
    assert answer.text == text
    assert answer.evidence_refs == []
    with pytest.raises(ValueError, match="completion"):
        validate_answer(AnswerCandidate(text=text), [], context(), [])


@pytest.mark.asyncio
async def test_side_question_preserves_accepted_requirements_and_pending_proposal():
    requirements, owner, conversation, proposal, _ = pending_proposal()
    trusted, catalog = context(requirements), Catalog()
    trusted.account_id, trusted.conversation_id = owner, conversation
    task = CompletionRequirement(kind="BUILD", constraints_revision=0)
    waiting = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Bạn xác nhận thay đổi RAM không?",
        waiting_tasks=[{"task": task, "status": "UNRESOLVED"}],
        pending_proposal=proposal,
    )
    accepted = AcceptedConversationSnapshot.model_validate(
        base_snapshot(trusted).model_dump() | {"flexible": waiting.model_dump()}
    )
    before = accepted.model_dump(mode="json")
    question = "Thay đổi RAM đang đề xuất là gì?"
    response = "Đề xuất RAM 16GB đang chờ bạn xác nhận, chưa được áp dụng."

    class ExplainProposal:
        async def decide(self, message, transcript, allowed, repair, context=None):
            assert context.requirements == requirements
            assert context.pending.proposal == proposal.patch
            assert context.pending.proposal_sources == list(
                proposal.source_messages.values()
            )
            return FlexibleDecision(action={"kind": "RESPOND", "text": response})

    flow = build_flexible_graph(
        ExplainProposal(),
        ToolDispatcher(catalog, PCBuildApplicationService()),
        trusted,
        [task],
        Budget(),
        conversation_revision=13,
        publication=FlexiblePublicationInput(
            accepted=accepted,
            task_message_id=waiting.task_message_id,
            task_message=waiting.task_message,
            request_message=question,
        ),
    )
    terminal = await flow.ainvoke({"message": question})
    published = AcceptedConversationSnapshot.model_validate(
        terminal["accepted_snapshot"]
    )
    assert published.flexible.assistant_message == response
    assert published.requirement_state == requirements
    assert published.public_build == accepted.public_build
    assert published.flexible.pending_proposal == proposal
    assert published.flexible.task_message_id == waiting.task_message_id
    assert published.flexible.task_outcomes == waiting.task_outcomes
    assert accepted.model_dump(mode="json") == before
    assert catalog.calls == 0


@pytest.mark.asyncio
async def test_empty_task_turn_publishes_response_and_is_revalidated():
    from ai_service.capabilities.assistant.react.snapshot_publication import (
        revalidate_flexible_publication,
    )
    from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
    from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1

    trusted, products, service = context(), Catalog(), PCBuildApplicationService()
    accepted = base_snapshot(trusted)
    request = TurnRequestV2(
        conversation_id=trusted.conversation_id,
        request_id=uuid4(),
        expected_revision=0,
        message="Chào bạn",
    )

    class Extractor:
        async def extract_tasks(self, message):
            return TaskDraftBatch(tasks=[])

    class Reply:
        async def decide(self, message, transcript, allowed, repair, context=None):
            assert context.completion_requirements == []
            return FlexibleDecision(action=RespondAction(text="Chào bạn!"))

    flow = build_flexible_graph(
        Reply(),
        ToolDispatcher(products, service),
        trusted,
        [],
        Budget(),
        conversation_revision=0,
        publication=FlexiblePublicationInput.for_request(accepted, request),
        task_extractor=Extractor(),
        task_catalog=products,
    )
    terminal = await flow.ainvoke({"message": request.message})
    restored = revalidate_flexible_publication(
        terminal,
        accepted,
        request,
        owner=trusted.account_id,
        run_id=trusted.run_id,
        base_ref=CheckpointRefV1(
            thread_id=trusted.conversation_id, checkpoint_id="root"
        ),
        service=service,
    )
    assert restored.flexible.assistant_message == "Chào bạn!"
    assert restored.flexible.task_outcomes == []
    assert restored.flexible.answer is None
    assert products.calls == 0
    forged = dict(terminal) | {"response": {"text": "forged", "evidence_refs": []}}
    with pytest.raises(ValueError, match="Response differs"):
        revalidate_flexible_publication(
            forged,
            accepted,
            request,
            owner=trusted.account_id,
            run_id=trusted.run_id,
            base_ref=CheckpointRefV1(
                thread_id=trusted.conversation_id, checkpoint_id="root"
            ),
            service=service,
        )


@pytest.mark.asyncio
async def test_grounded_conversation_response_keeps_product_cards_in_sse(monkeypatch):
    from ai_service.capabilities.assistant.react.contracts import ToolCall
    from ai_service.infrastructure.persistence import lifecycle_events

    trusted, products, service = context(), Catalog(), PCBuildApplicationService()
    call = ToolCall(call_id="read", name="search_catalog", arguments={"query": "GPU"})
    result = await ToolDispatcher(products, service).execute(call, trusted)
    snapshot = base_snapshot(trusted)
    snapshot.flexible = FlexiblePublishedState(
        conversation_id=trusted.conversation_id,
        task_message_id=uuid4(),
        task_message="GPU?",
        status="COMPLETED",
        response={"text": "Canonical GPU", "evidence_refs": result.evidence_refs},
    )
    events = []

    async def append(*args):
        events.append(args[-1])

    monkeypatch.setattr(lifecycle_events, "append_run_event", append)
    await lifecycle_events.emit_terminal(
        None,
        {
            "id": trusted.run_id,
            "conversation_id": trusted.conversation_id,
            "request_id": uuid4(),
            "status": "RUNNING",
            "base_revision": 0,
            "request_payload": {"schema_version": 2},
        },
        "COMPLETED",
        1,
        {
            "accepted_snapshot": snapshot.model_dump(mode="json"),
            "answer": snapshot.flexible.assistant_message,
            "ledger": [
                {
                    "call": call.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                }
            ],
        },
    )
    assert events[-1]["result"]["products"][0]["name"] == "Canonical GPU"
    assert events[-1]["result"]["flexible"]["task_outcomes"] == []


def test_conversation_response_cannot_complete_build_or_select_forged_evidence():
    trusted = context()
    with pytest.raises(ValueError, match="business result"):
        validate_response(
            RespondAction(text="Ưu tiên GPU"),
            trusted,
            [CompletionRequirement(kind="BUILD", constraints_revision=0)],
            [],
        )
    with pytest.raises(ValueError, match="evidence"):
        validate_response(
            RespondAction(text="Chào", evidence_refs=["made-up"]),
            trusted,
            [],
            [],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_proposal", [False, True])
async def test_requirements_can_be_collected_without_fabricating_a_build_task(
    agent_proposal,
):
    from test_explicit_user_admission import draft

    from ai_service.capabilities.assistant.react.snapshot_publication import (
        revalidate_flexible_publication,
    )
    from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
    from ai_service.capabilities.assistant.requirements.proposals import (
        ProposalConfirmation,
    )
    from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1

    trusted, products, service = context(), Catalog(), PCBuildApplicationService()
    base = base_snapshot(trusted)
    message = "Tư vấn giúp" if agent_proposal else "RAM 32GB"

    class Extractor:
        async def extract_tasks(self, message):
            return TaskDraftBatch(tasks=[])

        async def extract(self, message, requirements):
            return draft()

    class Model:
        def __init__(self):
            self.calls = 0

        async def decide(self, *args):
            self.calls += 1
            if self.calls == 1:
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "draft": draft().model_dump(),
                        "source_text": "RAM 32GB" if agent_proposal else None,
                    }
                )
            return FlexibleDecision(action=RespondAction(text="Đã ghi nhận RAM 32GB."))

    request = TurnRequestV2(
        conversation_id=trusted.conversation_id,
        request_id=uuid4(),
        expected_revision=0,
        message=message,
    )
    flow = build_flexible_graph(
        Model(),
        ToolDispatcher(products, service),
        trusted,
        [],
        Budget(),
        conversation_revision=0,
        publication=FlexiblePublicationInput.for_request(base, request),
        task_extractor=Extractor(),
        task_catalog=products,
        user_extractor=Extractor(),
    )
    terminal = await flow.ainvoke({"message": message})
    published = AcceptedConversationSnapshot.model_validate(
        terminal["accepted_snapshot"]
    )
    assert (
        revalidate_flexible_publication(
            terminal,
            base,
            request,
            owner=trusted.account_id,
            run_id=trusted.run_id,
            base_ref=CheckpointRefV1(
                thread_id=trusted.conversation_id, checkpoint_id="root"
            ),
            service=service,
        )
        == published
    )
    assert published.flexible.task_outcomes == []
    if agent_proposal:
        assert published.requirement_state == base.requirement_state
        proposal = published.flexible.pending_proposal

        class Acknowledge:
            async def decide(self, *args):
                return FlexibleDecision(
                    action=RespondAction(text="Đã ghi nhận RAM 32GB.")
                )

        confirmation = TurnRequestV2(
            conversation_id=trusted.conversation_id,
            request_id=uuid4(),
            expected_revision=1,
            message="Đồng ý",
            confirmation=ProposalConfirmation(
                proposal_id=proposal.proposal_id,
                proposal_hash=proposal.proposal_hash,
                target_revision=proposal.target_revision,
                base_requirements_hash=proposal.base_requirements_hash,
            ),
        )
        next_context = trusted.model_copy(update={"run_id": uuid4()}, deep=True)
        confirmed = await build_flexible_graph(
            Acknowledge(),
            ToolDispatcher(products, service),
            next_context,
            [],
            Budget(),
            conversation_revision=1,
            publication=FlexiblePublicationInput.for_request(published, confirmation),
        ).ainvoke({"message": confirmation.message})
        published = AcceptedConversationSnapshot.model_validate(
            confirmed["accepted_snapshot"]
        )
    assert published.flexible.task_outcomes == []
    assert published.flexible.pending_proposal is None
    assert published.flexible.response is not None
    assert len(published.requirement_state.requirements) == 1
    assert (
        next(iter(published.requirement_state.requirements.values())).value.amount == 32
    )
    assert products.calls == 0
