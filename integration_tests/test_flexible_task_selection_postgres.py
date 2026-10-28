"""Native selection/recovery gate; revisions are fixtures, not SQL publication."""

from uuid import uuid4

import pytest
from agent_task_proposals import WithTaskProposals
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from test_flexible_graph_postgres import Budget, Catalog

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_task_selection_reopen_preserves_source_and_selected_identity(
    postgres_dsn, unavailable
):
    class Products(Catalog):
        lookups = 0

        def __init__(self):
            super().__init__()
            self.first = self.product
            self.other = self.product.model_copy(
                update={
                    "product_id": uuid4(),
                    "name": "Alternative",
                    "evidence_ref": "catalog:other",
                }
            )
            self.second = self.product.model_copy(
                update={
                    "product_id": uuid4(),
                    "name": "Second",
                    "evidence_ref": "catalog:second",
                }
            )
            self.matches = {"GPU A": [self.first, self.other], "GPU B": [self.second]}

        async def resolve(self, mention):
            self.lookups += 1
            return self.matches.get(mention, [])

        async def get(self, identifier):
            self.calls += 1
            return next(
                (
                    p
                    for values in self.matches.values()
                    for p in values
                    if p.product_id == identifier
                ),
                None,
            )

    class Extractor:
        calls = 0

        async def extract_tasks(self, message):
            self.calls += 1
            return TaskDraftBatch(
                tasks=[
                    {
                        "kind": "COMPARE",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                        "subjects": [
                            {"start": 3, "end": 8, "quote": "GPU A"},
                            {"start": 12, "end": 17, "quote": "GPU B"},
                        ],
                    }
                ]
            )

    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Which GPU?",
                    "missing_fields": ["AMBIGUOUS"],
                }
            )

    class Continue:
        def __init__(self, fail):
            self.fail = fail

        async def decide(self, message, transcript, allowed, repair, context=None):
            if unavailable:
                if self.fail:
                    raise TimeoutError("after committed task resolution")
                return FlexibleDecision(
                    action={
                        "kind": "ASK_CLARIFICATION",
                        "question": "Product unavailable",
                        "missing_fields": ["SELECTED_PRODUCT_UNAVAILABLE"],
                    }
                )
            if not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "compare",
                            "name": "compare_products",
                            "arguments": {
                                "product_ids": [
                                    str(p)
                                    for p in context.completion_requirements[
                                        0
                                    ].product_ids
                                ]
                            },
                        },
                    }
                )
            if self.fail:
                raise TimeoutError("after committed comparison")
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "untrusted",
                        "evidence_refs": transcript[-1].result.evidence_refs,
                    },
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
    products, extractor, message = Products(), Extractor(), "So GPU A và GPU B"

    def compile_graph(model, binding, current, revision, saver):
        return build_flexible_graph(
            WithTaskProposals(model, extractor),
            ToolDispatcher(products, PCBuildApplicationService()),
            current,
            [],
            Budget(),
            conversation_revision=revision,
            publication=binding,
            task_catalog=products,
            checkpointer=saver,
        )

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(
                ctx.conversation_id
            )
            base = await PostgresAcceptedCheckpointReader(saver).read(root)
            binding = FlexiblePublicationInput(
                accepted=base,
                task_message_id=uuid4(),
                task_message=message,
                request_message=message,
            )
            flow = DurableConversationGraph(
                compile_graph(Ask(), binding, ctx, 0, saver),
                accepted_versions={FLEXIBLE_GRAPH_VERSION},
            )
            first = await flow.new_turn(
                ConversationHeadV1(
                    conversation_id=ctx.conversation_id,
                    revision=0,
                    accepted_checkpoint_ref=root,
                ),
                ctx.run_id,
                {"message": message},
                FLEXIBLE_GRAPH_VERSION,
            )
        current = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            accepted = await PostgresAcceptedCheckpointReader(saver).read(first)
            binding = FlexiblePublicationInput(
                accepted=accepted,
                task_message_id=accepted.flexible.task_message_id,
                task_message=message,
                request_message="Choose first",
                task_reply={
                    "pending_message_id": accepted.flexible.task_message_id,
                    "constraints_revision": 0,
                    "selections": [
                        {
                            "task_index": 0,
                            "subject_index": 0,
                            "product_id": products.first.product_id,
                        }
                    ],
                },
            )
            if unavailable:
                products.matches["GPU A"] = [products.other]
            graph = compile_graph(Continue(fail=True), binding, current, 1, saver)
            flow = DurableConversationGraph(
                graph, accepted_versions={FLEXIBLE_GRAPH_VERSION}
            )
            with pytest.raises(TimeoutError):
                await flow.new_turn(
                    ConversationHeadV1(
                        conversation_id=ctx.conversation_id,
                        revision=1,
                        accepted_checkpoint_ref=first,
                    ),
                    current.run_id,
                    {"message": "Choose first"},
                    FLEXIBLE_GRAPH_VERSION,
                )
            recovery = await graph.aget_state(
                {"configurable": {"thread_id": str(ctx.conversation_id)}}
            )
            assert recovery.next == ("agent",)
            # Latest is used only to locate this fixture's sole failed execution;
            # new-turn invocation above uses the exact accepted `first` ref.
            assert recovery.metadata["app_run_id"] == str(current.run_id)
            counts = (extractor.calls, products.lookups, products.calls)
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            graph = compile_graph(Continue(fail=False), binding, current, 1, saver)
            result = await graph.ainvoke(None, recovery.config, durability="sync")
            from ai_service.application.ports.stateful_conversation import (
                AcceptedConversationSnapshot,
            )

            snapshot = AcceptedConversationSnapshot.model_validate(
                result["accepted_snapshot"]
            )
            assert (
                snapshot.flexible.task_message_id == accepted.flexible.task_message_id
            )
            assert snapshot.flexible.task_message == message
            assert (extractor.calls, products.lookups, products.calls) == counts
            assert extractor.calls == 1
            if unavailable:
                pending = snapshot.flexible.pending_task_resolution
                assert (
                    pending.resolution.pending[0].reason
                    == "SELECTED_PRODUCT_UNAVAILABLE"
                )
                assert (
                    pending.resolution.bindings[0].product_id
                    == products.first.product_id
                )
                assert snapshot.flexible.status == "WAITING_INPUT"
            else:
                assert snapshot.flexible.status == "COMPLETED"
                assert snapshot.flexible.task_outcomes[0].task.product_ids == [
                    products.first.product_id,
                    products.second.product_id,
                ]
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))
