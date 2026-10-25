"""Native durable turns/revalidation; revisions are fixtures, not SQL CAS proof."""

from copy import deepcopy
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from test_flexible_graph_postgres import Budget, Catalog, Model

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    ProposalConfirmation,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.flexible import (
    FLEXIBLE_GRAPH_VERSION,
    build_flexible_graph,
)
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)


class Products(Catalog):
    async def resolve(self, mention):
        raise AssertionError("Conversation/proposal requires no product lookup")


class Extractor:
    def __init__(self, business=False):
        self.business = business

    async def extract_tasks(self, message):
        return TaskDraftBatch(
            tasks=[{"kind": "BUILD", "start": 0, "end": len(message), "quote": message}]
            if self.business
            else []
        )


class Respond:
    def __init__(self, text):
        self.text = text

    async def decide(self, *args):
        return FlexibleDecision(action={"kind": "RESPOND", "text": self.text})


class AskBudget:
    async def decide(self, *args):
        return FlexibleDecision(
            action={
                "kind": "ASK_CLARIFICATION",
                "question": "Bạn có ngân sách bao nhiêu?",
                "missing_fields": ["BUDGET_REQUIRED"],
            }
        )


def context(conversation_id):
    return ToolContext(
        account_id=uuid4(),
        conversation_id=conversation_id,
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )


async def execute_turn(dsn, ctx, base_ref, accepted, request, model, *, extractor=None):
    products = Products()
    tasks = (
        [row.task for row in accepted.flexible.task_outcomes]
        if accepted.flexible and extractor is None
        else []
    )
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        graph = build_flexible_graph(
            model,
            ToolDispatcher(products, PCBuildApplicationService()),
            ctx,
            tasks,
            Budget(),
            conversation_revision=request.expected_revision,
            publication=FlexiblePublicationInput.for_request(
                accepted, request, owner=ctx.account_id
            ),
            task_extractor=extractor,
            task_catalog=products if extractor else None,
            checkpointer=saver,
        )
        flow = DurableConversationGraph(
            graph, accepted_versions={FLEXIBLE_GRAPH_VERSION}
        )
        terminal = await flow.new_turn(
            ConversationHeadV1(
                conversation_id=ctx.conversation_id,
                revision=request.expected_revision,
                accepted_checkpoint_ref=base_ref,
            ),
            ctx.run_id,
            {"message": request.message},
            FLEXIBLE_GRAPH_VERSION,
        )
        assert products.calls == 0
        return terminal


async def reopen_and_revalidate(dsn, ref, accepted, request, ctx, base_ref):
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        record = await saver.aget_tuple(DurableConversationGraph.config(ref))
        values = record.checkpoint["channel_values"]
        restored = await PostgresAcceptedCheckpointReader(saver).read(ref)
        assert revalidate(values, accepted, request, ctx, base_ref) == restored
        return restored, values


def revalidate(values, accepted, request, ctx, base_ref):
    return revalidate_flexible_publication(
        values,
        accepted,
        request,
        owner=ctx.account_id,
        run_id=ctx.run_id,
        base_ref=base_ref,
        service=PCBuildApplicationService(),
    )


@pytest.mark.asyncio
async def test_no_business_task_response_survives_postgres_reopen(postgres_dsn):
    ctx = context(uuid4())
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=0,
        message="Chào bạn",
    )
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            accepted = await PostgresAcceptedCheckpointReader(saver).read(root)
        text = "Chào bạn! Bạn đang muốn tìm linh kiện hay tư vấn PC?"
        terminal = await execute_turn(
            postgres_dsn,
            ctx,
            root,
            accepted,
            request,
            Respond(text),
            extractor=Extractor(),
        )
        restored, values = await reopen_and_revalidate(
            postgres_dsn, terminal, accepted, request, ctx, root
        )
        assert restored.flexible.assistant_message == text
        assert restored.flexible.answer is None
        assert restored.flexible.task_outcomes == []
        assert restored.requirement_state == accepted.requirement_state
        assert values["task_draft"]["tasks"] == []
        for target in ("response", "accepted_snapshot"):
            corrupted = deepcopy(values)
            if target == "response":
                corrupted["response"]["text"] = "Injected response"
            else:
                corrupted["accepted_snapshot"]["flexible"]["response"]["text"] = (
                    "Injected persisted response"
                )
            with pytest.raises(ValueError):
                revalidate(corrupted, accepted, request, ctx, root)
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
async def test_proposal_side_response_then_original_confirmation_after_reopen(
    postgres_dsn,
):
    ctx = context(uuid4())
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            initial = await PostgresAcceptedCheckpointReader(saver).read(root)
        original = TurnRequestV2(
            conversation_id=ctx.conversation_id,
            request_id=uuid4(),
            expected_revision=0,
            message="Build PC",
        )
        first_ref = await execute_turn(
            postgres_dsn,
            ctx,
            root,
            initial,
            original,
            Model(propose=True),
            extractor=Extractor(business=True),
        )
        pending, _ = await reopen_and_revalidate(
            postgres_dsn, first_ref, initial, original, ctx, root
        )
        proposal = pending.flexible.pending_proposal
        assert proposal.target_revision == 1
        question = TurnRequestV2(
            conversation_id=ctx.conversation_id,
            request_id=uuid4(),
            expected_revision=1,
            message="Nâng RAM như vậy có lợi gì?",
        )
        side_ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
        text = "RAM đang đề xuất chưa được áp dụng; bạn có thể cân nhắc theo nhu cầu."
        side_ref = await execute_turn(
            postgres_dsn, side_ctx, first_ref, pending, question, Respond(text)
        )
        side, values = await reopen_and_revalidate(
            postgres_dsn, side_ref, pending, question, side_ctx, first_ref
        )
        assert side.flexible.assistant_message == text
        assert side.flexible.status == "WAITING_INPUT"
        assert side.flexible.pending_proposal == proposal
        assert side.flexible.task_message_id == original.request_id
        assert side.flexible.task_outcomes == pending.flexible.task_outcomes
        assert side.requirement_state == initial.requirement_state
        for target in ("pending_proposal", "accepted_snapshot"):
            corrupted = deepcopy(values)
            if target == "pending_proposal":
                corrupted["pending_proposal"]["target_revision"] = 2
            else:
                corrupted["accepted_snapshot"]["flexible"]["pending_proposal"] = None
            with pytest.raises(ValueError):
                revalidate(corrupted, pending, question, side_ctx, first_ref)
        confirmation = ProposalConfirmation(
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.proposal_hash,
            target_revision=proposal.target_revision,
            base_requirements_hash=proposal.base_requirements_hash,
        )
        confirm_request = TurnRequestV2(
            conversation_id=ctx.conversation_id,
            request_id=uuid4(),
            expected_revision=2,
            message="Đồng ý",
            confirmation=confirmation,
        )
        confirm_ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
        final_ref = await execute_turn(
            postgres_dsn, confirm_ctx, side_ref, side, confirm_request, AskBudget()
        )
        confirmed, _ = await reopen_and_revalidate(
            postgres_dsn, final_ref, side, confirm_request, confirm_ctx, side_ref
        )
        assert confirmed.flexible.pending_proposal is None
        assert confirmed.flexible.clarification.missing_fields == ["BUDGET_REQUIRED"]
        assert len(confirmed.requirement_state.requirements) == 1
        assert confirmed.requirement_state.revision == 1
        applied = next(iter(confirmed.requirement_state.requirements.values()))
        assert applied.value.amount == 32
        duplicate_ctx = confirm_ctx.model_copy(
            update={"run_id": uuid4(), "requirements": confirmed.requirement_state},
            deep=True,
        )
        duplicate = confirm_request.model_copy(
            update={"request_id": uuid4(), "expected_revision": 3}, deep=True
        )
        with pytest.raises(ValueError, match="[Pp]roposal"):
            await execute_turn(
                postgres_dsn,
                duplicate_ctx,
                final_ref,
                confirmed,
                duplicate,
                AskBudget(),
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            assert (
                await PostgresAcceptedCheckpointReader(saver).read(final_ref)
                == confirmed
            )
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))
