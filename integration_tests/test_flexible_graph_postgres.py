"""V2 checkpoint gate; model/catalog/attempts are fixtures, no publication claim."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ProductEvidence,
    ToolContext,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


class Budget:
    async def invoke(self, kind, operation_key, request_hash, dispatch):
        return await dispatch()


class Catalog:
    def __init__(self):
        self.calls = 0
        self.product = ProductEvidence(
            product_id=uuid4(),
            name="GPU",
            price_vnd=8_000_000,
            retrieved_at=datetime.now(UTC),
            evidence_ref="catalog:gpu",
        )

    async def search(self, query, limit):
        self.calls += 1
        return [self.product]

    async def get(self, product_id):
        return self.product if product_id == self.product.product_id else None


class Model:
    def __init__(self, fail_after_tool=False, propose=False):
        self.fail_after_tool, self.propose = fail_after_tool, propose

    async def decide(self, message, transcript, allowed, repair, context=None):
        if self.propose:
            return FlexibleDecision(
                action={
                    "kind": "PROPOSE_REQUIREMENTS",
                    "source_text": "RAM 32GB",
                    "draft": {
                        "items": [
                            {
                                "subject": "RAM",
                                "predicate": "capacity",
                                "start": 0,
                                "end": 8,
                                "value": {
                                    "kind": "QUANTITY",
                                    "amount": 32,
                                    "unit": "GB",
                                },
                            }
                        ]
                    },
                }
            )
        if not transcript:
            return FlexibleDecision(
                action={
                    "kind": "CALL_TOOL",
                    "call": {
                        "call_id": "search",
                        "name": "search_catalog",
                        "arguments": {"query": "GPU"},
                    },
                }
            )
        if self.fail_after_tool:
            raise TimeoutError("failure after committed read")
        return FlexibleDecision(
            action={
                "kind": "FINAL_ANSWER",
                "answer": {
                    "text": "unverified",
                    "evidence_refs": transcript[-1].result.evidence_refs,
                },
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("proposal", [False, True])
async def test_v2_reopen_preserves_tool_or_confirmation_boundary(
    postgres_dsn, proposal
):
    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    catalog = Catalog()
    dispatcher = ToolDispatcher(catalog, PCBuildApplicationService())
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}

    def compile_flow(model, saver, revision=0):
        return build_flexible_graph(
            model,
            dispatcher,
            ctx,
            [CompletionRequirement(kind="ANSWER")],
            Budget(),
            conversation_revision=revision,
            checkpointer=saver,
        )

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            flow = compile_flow(Model(fail_after_tool=True, propose=proposal), saver)
            if proposal:
                first = await flow.ainvoke(
                    {"message": "RAM"}, config, durability="sync"
                )
                assert first["phase"] == "WAITING_INPUT"
            else:
                with pytest.raises(TimeoutError):
                    await flow.ainvoke({"message": "GPU"}, config, durability="sync")
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            flow = compile_flow(Model(), saver, revision=1 if proposal else 0)
            snapshot = await flow.aget_state(config)
            if proposal:
                pending = PendingRequirementProposal.model_validate(
                    snapshot.values["pending_proposal"]
                )
                assert pending.target_revision == 1
                reply = ProposalConfirmation(
                    proposal_id=pending.proposal_id,
                    proposal_hash=pending.proposal_hash,
                    target_revision=1,
                    base_requirements_hash=pending.base_requirements_hash,
                )
                result = await flow.ainvoke(
                    {"message": "OK", "confirmation": reply.model_dump(mode="json")},
                    snapshot.config,
                    durability="sync",
                )
                assert len(result["working_requirements"]["requirements"]) == 1
                assert result["pending_proposal"] is None
            else:
                assert snapshot.next == ("agent",)
                result = await flow.ainvoke(None, snapshot.config, durability="sync")
            assert result["phase"] == "READY_TO_PUBLISH" and catalog.calls == 1
            assert ctx.requirements.requirements == {}
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
async def test_confirm_then_clarify_preparation_survives_reopen(postgres_dsn):
    """Checkpoint/preparation chain; head revisions are fixtures, not SQL CAS proof."""
    from ai_service.application.ports.stateful_conversation import (
        AcceptedConversationSnapshot,
    )
    from ai_service.capabilities.assistant.react.waiting_publication import (
        prepare_waiting_publication,
    )
    from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
    from ai_service.infrastructure.graph.durable import DurableConversationGraph
    from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION
    from ai_service.infrastructure.graph.initial_root import (
        NativeConversationInitializer,
    )

    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Ngân sách?",
                    "missing_fields": ["BUDGET_REQUIRED"],
                }
            )

    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            flow = DurableConversationGraph(
                build_flexible_graph(
                    Model(propose=True),
                    dispatcher,
                    ctx,
                    tasks,
                    Budget(),
                    conversation_revision=0,
                    checkpointer=saver,
                ),
                accepted_versions={FLEXIBLE_GRAPH_VERSION},
            )
            head = ConversationHeadV1(
                conversation_id=ctx.conversation_id,
                revision=0,
                accepted_checkpoint_ref=root,
            )
            first = await flow.new_turn(
                head, ctx.run_id, {"message": "Tư vấn"}, FLEXIBLE_GRAPH_VERSION
            )
            root_record = await saver.aget_tuple(DurableConversationGraph.config(root))
            base = AcceptedConversationSnapshot.model_validate(
                root_record.checkpoint["channel_values"]
            )
            first_values = (await flow.read(first)).values
            waiting = prepare_waiting_publication(
                first_values,
                base,
                ctx,
                tasks,
                source_message="Tư vấn",
                conversation_revision=0,
            )
            pending = waiting.pending_proposal
        current = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
        confirmation = ProposalConfirmation(
            proposal_id=pending.proposal_id,
            proposal_hash=pending.proposal_hash,
            target_revision=1,
            base_requirements_hash=pending.base_requirements_hash,
        )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            flow = DurableConversationGraph(
                build_flexible_graph(
                    Ask(),
                    dispatcher,
                    current,
                    tasks,
                    Budget(),
                    conversation_revision=1,
                    checkpointer=saver,
                ),
                accepted_versions={FLEXIBLE_GRAPH_VERSION},
            )
            head = ConversationHeadV1(
                conversation_id=ctx.conversation_id,
                revision=1,
                accepted_checkpoint_ref=first,
            )
            second = await flow.new_turn(
                head,
                current.run_id,
                {"message": "OK", "confirmation": confirmation.model_dump(mode="json")},
                FLEXIBLE_GRAPH_VERSION,
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            values = (
                await saver.aget_tuple(DurableConversationGraph.config(second))
            ).checkpoint["channel_values"]
            prepared = prepare_waiting_publication(
                values,
                waiting.snapshot,
                current,
                tasks,
                source_message="OK",
                conversation_revision=1,
                accepted_proposal=pending,
                confirmation=confirmation,
            )
            assert len(prepared.snapshot.requirement_state.requirements) == 1
            assert prepared.clarification.missing_fields == ["BUDGET_REQUIRED"]
            assert prepared.task_outcomes[0].status == "UNRESOLVED"
            assert waiting.snapshot.requirement_state.requirements == {}
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("unresolved", [False, True])
async def test_new_waiting_preparation_revalidates_artifact_after_reopen(
    postgres_dsn, unresolved
):
    from ai_service.application.ports.stateful_conversation import (
        AcceptedConversationSnapshot,
    )
    from ai_service.capabilities.assistant.react.waiting_publication import (
        prepare_waiting_publication,
    )
    from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
    from ai_service.infrastructure.graph.durable import DurableConversationGraph
    from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION
    from ai_service.infrastructure.graph.initial_root import (
        NativeConversationInitializer,
    )

    class WaitingModel(Model):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if unresolved:
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "source_text": "RAM 32",
                        "draft": {
                            "items": [
                                {
                                    "subject": "RAM",
                                    "predicate": "capacity",
                                    "start": 0,
                                    "end": 6,
                                    "value": {
                                        "kind": "UNRESOLVED_QUANTITY",
                                        "amount": 32,
                                    },
                                }
                            ]
                        },
                    }
                )
            return await super().decide(message, transcript, allowed, repair, context)

    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    message = "Tư vấn RAM"
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            compiled = build_flexible_graph(
                WaitingModel(propose=True),
                ToolDispatcher(Catalog(), PCBuildApplicationService()),
                ctx,
                tasks,
                Budget(),
                conversation_revision=0,
                checkpointer=saver,
            )
            flow = DurableConversationGraph(
                compiled, accepted_versions={FLEXIBLE_GRAPH_VERSION}
            )
            head = ConversationHeadV1(
                conversation_id=ctx.conversation_id,
                revision=0,
                accepted_checkpoint_ref=root,
            )
            terminal = await flow.new_turn(
                head, ctx.run_id, {"message": message}, FLEXIBLE_GRAPH_VERSION
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            accepted = await saver.aget_tuple(DurableConversationGraph.config(root))
            base = AcceptedConversationSnapshot.model_validate(
                accepted.checkpoint["channel_values"]
            )
            record = await saver.aget_tuple(DurableConversationGraph.config(terminal))
            prepared = prepare_waiting_publication(
                record.checkpoint["channel_values"],
                base,
                ctx,
                tasks,
                source_message=message,
                conversation_revision=0,
            )
            assert prepared.status == "WAITING_INPUT" and prepared.snapshot == base
            assert (prepared.pending_resolution is not None) == unresolved
            assert all(task.status == "UNRESOLVED" for task in prepared.task_outcomes)
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
async def test_answer_preparation_revalidates_reopened_terminal_without_publishing(
    postgres_dsn,
):
    from ai_service.application.ports.stateful_conversation import (
        AcceptedConversationSnapshot,
    )
    from ai_service.capabilities.assistant.react.publication import (
        prepare_answer_publication,
    )
    from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
    from ai_service.infrastructure.graph.durable import DurableConversationGraph
    from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION
    from ai_service.infrastructure.graph.initial_root import (
        NativeConversationInitializer,
    )

    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    service = PCBuildApplicationService()
    dispatcher = ToolDispatcher(Catalog(), service)
    tasks = [CompletionRequirement(kind="ANSWER")]
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            compiled = build_flexible_graph(
                Model(),
                dispatcher,
                ctx,
                tasks,
                Budget(),
                conversation_revision=0,
                checkpointer=saver,
            )
            flow = DurableConversationGraph(
                compiled, accepted_versions={FLEXIBLE_GRAPH_VERSION}
            )
            head = ConversationHeadV1(
                conversation_id=ctx.conversation_id,
                revision=0,
                accepted_checkpoint_ref=root,
            )
            terminal = await flow.new_turn(
                head, ctx.run_id, {"message": "GPU"}, FLEXIBLE_GRAPH_VERSION
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            accepted = await saver.aget_tuple(DurableConversationGraph.config(root))
            base = AcceptedConversationSnapshot.model_validate(
                accepted.checkpoint["channel_values"]
            )
            record = await saver.aget_tuple(DurableConversationGraph.config(terminal))
            values = record.checkpoint["channel_values"]
            before = await saver.aget_tuple(
                {"configurable": {"thread_id": str(ctx.conversation_id)}}
            )
            prepared = prepare_answer_publication(values, base, ctx, tasks, service)
            assert (
                prepared.answer.status == "COMPLETED" and "GPU" in prepared.answer.text
            )
            assert prepared.snapshot.model_dump(mode="json") == base.model_dump(
                mode="json"
            )
            after = await saver.aget_tuple(
                {"configurable": {"thread_id": str(ctx.conversation_id)}}
            )
            assert before.config == after.config
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
async def test_v2_native_wrapper_uses_accepted_head_and_rejects_legacy_after_reopen(
    postgres_dsn,
):
    from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
    from ai_service.infrastructure.graph.durable import DurableConversationGraph
    from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION
    from ai_service.infrastructure.graph.initial_root import (
        NativeConversationInitializer,
    )

    class BranchModel(Model):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if message == "orphan" and len(transcript) == 1:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "orphan",
                            "name": "search_catalog",
                            "arguments": {"query": "GPU"},
                        },
                    }
                )
            return await super().decide(message, transcript, allowed, repair, context)

    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    catalog = Catalog()
    dispatcher = ToolDispatcher(catalog, PCBuildApplicationService())

    def compile_flow(saver):
        return build_flexible_graph(
            BranchModel(),
            dispatcher,
            ctx,
            [CompletionRequirement(kind="ANSWER")],
            Budget(),
            conversation_revision=0,
            checkpointer=saver,
        )

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            flow = DurableConversationGraph(
                compile_flow(saver), accepted_versions={FLEXIBLE_GRAPH_VERSION}
            )
            head = ConversationHeadV1(
                conversation_id=ctx.conversation_id,
                revision=0,
                accepted_checkpoint_ref=root,
            )
            accepted = await flow.new_turn(
                head, uuid4(), {"message": "first"}, FLEXIBLE_GRAPH_VERSION
            )
            head = head.model_copy(
                update={"revision": 1, "accepted_checkpoint_ref": accepted}
            )
            orphan = await flow.new_turn(
                head, uuid4(), {"message": "orphan"}, FLEXIBLE_GRAPH_VERSION
            )
            assert len((await flow.read(orphan)).values["ledger"]) == 2
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            compiled = compile_flow(saver)
            flow = DurableConversationGraph(
                compiled, accepted_versions={FLEXIBLE_GRAPH_VERSION}
            )
            final = await flow.new_turn(
                head, uuid4(), {"message": "third"}, FLEXIBLE_GRAPH_VERSION
            )
            snapshot = await flow.read(final)
            assert [row["call"]["call_id"] for row in snapshot.values["ledger"]] == [
                "search"
            ]
            assert snapshot.metadata["app_base_checkpoint_id"] == accepted.checkpoint_id
            raw = DurableConversationGraph(compiled)
            legacy = await raw.new_turn(
                head, uuid4(), {"message": "legacy"}, "legacy-guided-v1"
            )
            incompatible = head.model_copy(
                update={"revision": 2, "accepted_checkpoint_ref": legacy}
            )
            before = await saver.aget_tuple(
                {"configurable": {"thread_id": str(ctx.conversation_id)}}
            )
            with pytest.raises(ValueError, match="version"):
                await flow.new_turn(
                    incompatible,
                    uuid4(),
                    {"message": "blocked"},
                    FLEXIBLE_GRAPH_VERSION,
                )
            after = await saver.aget_tuple(
                {"configurable": {"thread_id": str(ctx.conversation_id)}}
            )
            assert before.config == after.config
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [False, True])
async def test_typed_choice_after_postgres_reopen_still_requires_confirmation(
    postgres_dsn,
    unit,
):
    from ai_service.application.ports.hardware import ComponentSpec
    from ai_service.capabilities.assistant.provenance import CatalogRowV1
    from ai_service.capabilities.assistant.requirements.pending_resolution import (
        PendingResolution,
    )
    from ai_service.capabilities.assistant.stateful_contracts import (
        ClarificationReplyV1,
        UnitClarificationReplyV1,
    )

    catalog_rows = [
        CatalogRowV1(
            variant_id=(identifier := uuid4()),
            sku=f"GPU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=ComponentSpec(
                id=identifier,
                name=f"RTX 5070 Ti {index}",
                category="GPU",
                price=20_000_000,
            ),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["catalog:gpu"],
        )
        for index in range(2)
    ]
    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        catalog=catalog_rows,
        read_catalog_allowed=True,
        execution_allowed=True,
    )

    class Proposer:
        async def decide(self, *args):
            if unit:
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "source_text": "SSD 1",
                        "draft": {
                            "items": [
                                {
                                    "subject": "STORAGE",
                                    "predicate": "capacity",
                                    "start": 0,
                                    "end": 5,
                                    "value": {
                                        "kind": "UNRESOLVED_QUANTITY",
                                        "amount": 1,
                                    },
                                }
                            ]
                        },
                    }
                )
            return FlexibleDecision(
                action={
                    "kind": "PROPOSE_REQUIREMENTS",
                    "source_text": "RTX 5070 Ti",
                    "draft": {
                        "items": [
                            {
                                "subject": "GPU",
                                "predicate": "identity",
                                "start": 0,
                                "end": 11,
                                "value": {"kind": "TEXT", "text": "RTX 5070 Ti"},
                            }
                        ]
                    },
                }
            )

    class NoModel:
        async def decide(self, *args):
            raise AssertionError("Typed choice must not use model after reopen")

    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            flow = build_flexible_graph(
                Proposer(),
                dispatcher,
                ctx,
                tasks,
                Budget(),
                conversation_revision=0,
                checkpointer=saver,
            )
            result = await flow.ainvoke(
                {"message": "Tư vấn GPU"}, config, durability="sync"
            )
            assert result["phase"] == "WAITING_INPUT"
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            flow = build_flexible_graph(
                NoModel(),
                dispatcher,
                ctx,
                tasks,
                Budget(),
                conversation_revision=1,
                checkpointer=saver,
            )
            snapshot = await flow.aget_state(config)
            pending = PendingResolution.model_validate(snapshot.values["pending_draft"])
            if unit:
                unit_reply = UnitClarificationReplyV1(
                    pending_message_id=pending.draft.message_id,
                    constraints_revision=0,
                    selections=[
                        {
                            "item_index": 0,
                            "unit": "TB",
                            "quote": "TB",
                            "start": 5,
                            "end": 7,
                        }
                    ],
                )
                command = {
                    "message": "Dùng TB nhé",
                    "unit_reply": unit_reply.model_dump(mode="json"),
                }
            else:
                reply = ClarificationReplyV1(
                    pending_message_id=pending.draft.message_id,
                    selections=[
                        {"item_index": 0, "variant_id": catalog_rows[1].variant_id}
                    ],
                )
                command = {
                    "message": "Chọn thứ hai",
                    "reference_reply": reply.model_dump(mode="json"),
                }
            result = await flow.ainvoke(
                command,
                snapshot.config,
                durability="sync",
            )
            assert (
                result["phase"] == "WAITING_INPUT" and result["pending_draft"] is None
            )
            proposal = PendingRequirementProposal.model_validate(
                result["pending_proposal"]
            )
            assert proposal.target_revision == 2
            if unit:
                assert proposal.patch.set_requirements[0].value.amount == 1000
                assert "Dùng TB nhé" in proposal.source_messages.values()
            else:
                assert (
                    proposal.patch.set_requirements[0].value.variant_id
                    == catalog_rows[1].variant_id
                )
            assert result["working_requirements"]["requirements"] == {}
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))


@pytest.mark.asyncio
async def test_scoped_search_report_survives_reopen_without_reoptimizing(postgres_dsn):
    from ai_service.capabilities.assistant.requirements.contracts import (
        Evidence,
        Requirement,
    )

    source = uuid4()
    evidence = Evidence(message_id=source, start=0, end=1, quote="x")
    budget = Requirement(
        id=uuid4(),
        subject="SETUP",
        predicate="budget",
        operator="LTE",
        value={"kind": "QUANTITY", "amount": 1, "unit": "VND"},
        evidence=evidence,
    )
    goal = Requirement(
        id=uuid4(),
        subject="WORKLOAD",
        predicate="use_case",
        value={"kind": "TEXT", "text": "GAMING_1080P"},
        evidence=evidence,
    )
    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(
            requirements={budget.id: budget}, goals={goal.id: goal}
        ),
        read_catalog_allowed=True,
        execution_allowed=True,
        catalog=[],
    )

    class CountingService(PCBuildApplicationService):
        calls = 0

        def build_pc(self, *args, **kwargs):
            self.calls += 1
            return super().build_pc(*args, **kwargs)

    class SearchModel:
        def __init__(self, fail=False):
            self.fail = fail

        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "build",
                            "name": "build_pc",
                            "arguments": {},
                        },
                    }
                )
            if self.fail:
                raise TimeoutError("failed after optimizer checkpoint")
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "INFEASIBLE",
                    "supporting_tool_call_ids": ["build"],
                    "answer": {"text": "No solution globally"},
                }
            )

    service = CountingService()
    dispatcher = ToolDispatcher(Catalog(), service)
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    required = [CompletionRequirement(kind="BUILD", constraints_revision=0)]

    def compile_flow(model, saver):
        return build_flexible_graph(
            model,
            dispatcher,
            ctx,
            required,
            Budget(),
            conversation_revision=0,
            checkpointer=saver,
        )

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            flow = compile_flow(SearchModel(fail=True), saver)
            with pytest.raises(TimeoutError):
                await flow.ainvoke({"message": "Build PC"}, config, durability="sync")
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            flow = compile_flow(SearchModel(), saver)
            snapshot = await flow.aget_state(config)
            assert snapshot.next == ("agent",)
            result = await flow.ainvoke(None, snapshot.config, durability="sync")
            assert result["answer"]["status"] == "COMPLETED"
            assert (
                result["answer"]["search_evidence"]["scope"]
                == "SUPPLIED_NORMALIZED_CANDIDATES"
            )
            assert "globally" not in result["answer"]["text"] and service.calls == 1
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))
