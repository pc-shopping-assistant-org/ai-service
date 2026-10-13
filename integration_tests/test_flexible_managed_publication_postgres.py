"""V2 managed stop/lineage/CAS with SQL and native saver; provider/catalog fixtures."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine
from test_flexible_graph_postgres import Budget, Catalog, Model

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.recovery_policy import RetryFailure
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ExecutionStopEvidenceV1,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import (
    ExecutionFailed,
    ExecutionOutcome,
    ManagedExecutor,
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
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import (
    PostgresRunCoordinator,
    RunAdmissionError,
)
from ai_service.infrastructure.persistence.schema import (
    agent_call_attempts,
    agent_runs,
    conversation_messages,
    conversations,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "waiting,tamper,recover,reconcile",
    [
        (False, False, False, None),
        (True, False, False, None),
        (False, True, False, None),
        (False, False, True, None),
        (False, False, True, "MISSING"),
        (False, False, True, "STALE"),
        (False, False, True, "BRANCH"),
    ],
)
async def test_v2_managed_run_publishes_snapshot_message_and_head_once(
    postgres_dsn, waiting, tamper, recover, reconcile
):
    class Products(Catalog):
        async def resolve(self, mention):
            return []

    class Extractor:
        async def extract_tasks(self, message):
            return TaskDraftBatch(
                tasks=[
                    {
                        "kind": "BUILD" if waiting else "ANSWER",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, conversation, executor = (
        PostgresConversationStore(engine),
        uuid4(),
        None,
        None,
    )
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            conversation = await store.create(owner, "V2 fixture")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            service = PCBuildApplicationService()
            coordinator = PostgresRunCoordinator(
                engine,
                saver,
                request_versions=(2,),
                flexible_service=service,
                max_attempts=2 if recover else 1,
            )
            reader = PostgresAcceptedCheckpointReader(saver)
            compiled = {}
            failed_claims = []
            product_instances = []

            class RecoverableModel(Model):
                async def decide(self, *args, **kwargs):
                    if recover and args[1] and not failed_claims:
                        raise ExecutionFailed(RetryFailure.UPSTREAM_TIMEOUT)
                    return await super().decide(*args, **kwargs)

            async def factory(claim):
                request, head = await coordinator.load(claim)
                accepted = await reader.read(head.accepted_checkpoint_ref)
                ctx = ToolContext(
                    account_id=await coordinator.owner(claim),
                    conversation_id=claim.conversation_id,
                    run_id=claim.run_id,
                    requirements=accepted.requirement_state,
                    read_catalog_allowed=True,
                    execution_allowed=True,
                )
                products = Products()
                product_instances.append(products)
                graph = build_flexible_graph(
                    RecoverableModel(propose=waiting),
                    ToolDispatcher(products, service),
                    ctx,
                    [],
                    AttemptBoundary(coordinator.ledger, claim) if recover else Budget(),
                    conversation_revision=head.revision,
                    publication=FlexiblePublicationInput.for_request(accepted, request),
                    task_extractor=Extractor(),
                    task_catalog=products,
                    checkpointer=saver,
                )
                compiled[claim.run_id] = graph
                return DurableConversationGraph(
                    graph, accepted_versions={FLEXIBLE_GRAPH_VERSION}
                )

            if tamper:
                stage = coordinator.stage

                async def wrong_generation(claim, ref):
                    native = await saver.aget_tuple(
                        DurableConversationGraph.config(ref)
                    )
                    config = DurableConversationGraph.config(ref)
                    config["metadata"] = dict(native.metadata) | {
                        "app_generation": claim.generation + 1
                    }
                    updated = await compiled[claim.run_id].aupdate_state(
                        config,
                        {
                            "accepted_snapshot": native.checkpoint["channel_values"][
                                "accepted_snapshot"
                            ]
                        },
                        as_node="prepare_publish",
                    )
                    wrong = CheckpointRefV1(
                        thread_id=claim.conversation_id,
                        checkpoint_id=updated["configurable"]["checkpoint_id"],
                    )
                    with pytest.raises(ValueError, match="Working checkpoint identity"):
                        await coordinator.observe_checkpoint(
                            claim, wrong, FLEXIBLE_GRAPH_VERSION
                        )
                    await stage(claim, wrong)

                coordinator.stage = wrong_generation

            execution = ConversationExecution(
                coordinator, factory, FLEXIBLE_GRAPH_VERSION
            )

            async def track_execution(claim, children):
                try:
                    await execution(claim, children)
                except BaseException:
                    failed_claims.append(claim)
                    raise

            executor = ManagedExecutor(
                coordinator,
                track_execution,
                worker_id=uuid4(),
                poll_seconds=0.02,
            )
            await executor.start()
            request = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="Build PC" if waiting else "GPU?",
            )
            submitted = await coordinator.submit(owner, request)
            executor.wake()
            expected = (
                "FAILED_RETRYABLE"
                if recover
                else "WAITING_INPUT"
                if waiting
                else "COMPLETED"
            )
            for _ in range(150):
                view = await coordinator.get(owner, conversation.id, submitted.run_id)
                if view.status == expected or executor.last_error:
                    break
                await asyncio.sleep(0.02)
            record = await store.get(owner, conversation.id)
            if recover:
                assert view.status == "FAILED_RETRYABLE"
                assert record.revision == 0 and record.active_run_id == submitted.run_id
                previous = failed_claims[0]
                async with engine.connect() as conn:
                    original_attempts = set(
                        (
                            await conn.execute(
                                select(agent_call_attempts.c.id).where(
                                    agent_call_attempts.c.run_id == submitted.run_id
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                assert original_attempts
                if reconcile:
                    from sqlalchemy import update

                    entries = [
                        entry
                        async for entry in saver.alist(
                            {
                                "configurable": {
                                    "thread_id": str(conversation.id),
                                    "checkpoint_ns": "",
                                }
                            },
                            filter={
                                "app_run_id": str(previous.run_id),
                                "app_generation": previous.generation,
                            },
                        )
                    ]
                    original_ref = CheckpointRefV1(
                        thread_id=conversation.id,
                        checkpoint_id=entries[0].config["configurable"][
                            "checkpoint_id"
                        ],
                    )
                    if reconcile == "BRANCH":
                        for _ in range(2):
                            branch_config = DurableConversationGraph.config(
                                original_ref
                            )
                            branch_config["metadata"] = dict(entries[0].metadata)
                            await compiled[previous.run_id].aupdate_state(
                                branch_config,
                                {"repair_feedback": None},
                                as_node="execute_tool",
                            )
                        with pytest.raises(ValueError, match="ambiguous"):
                            await coordinator.claim_recovery(previous, uuid4())
                        current = await store.get(owner, conversation.id)
                        assert (
                            current.revision == 0
                            and current.active_run_id == submitted.run_id
                        )
                        async with engine.connect() as conn:
                            unchanged = (
                                (
                                    await conn.execute(
                                        select(agent_runs).where(
                                            agent_runs.c.id == submitted.run_id
                                        )
                                    )
                                )
                                .mappings()
                                .one()
                            )
                        assert unchanged["lease_generation"] == previous.generation
                        assert unchanged["authorized_recovery_source_refs"] == []
                        return
                    orphan_config = DurableConversationGraph.config(original_ref)
                    orphan_config["metadata"] = dict(entries[0].metadata) | {
                        "app_run_id": str(uuid4())
                    }
                    await compiled[previous.run_id].aupdate_state(
                        orphan_config, {"repair_feedback": None}, as_node="execute_tool"
                    )
                    stale = (
                        None
                        if reconcile == "MISSING"
                        else CheckpointRefV1(
                            thread_id=conversation.id,
                            checkpoint_id=entries[-1].config["configurable"][
                                "checkpoint_id"
                            ],
                        ).model_dump(mode="json")
                    )
                    async with engine.begin() as conn:
                        await conn.execute(
                            update(agent_runs)
                            .where(agent_runs.c.id == submitted.run_id)
                            .values(last_checkpoint_ref=stale)
                        )
                recovered = await coordinator.claim_recovery(previous, uuid4())
                assert recovered.generation == previous.generation + 1
                assert recovered.run_id == previous.run_id
                with pytest.raises(RunAdmissionError):
                    await coordinator.claim_recovery(previous, uuid4())
                async with engine.connect() as conn:
                    row = (
                        (
                            await conn.execute(
                                select(agent_runs).where(
                                    agent_runs.c.id == submitted.run_id
                                )
                            )
                        )
                        .mappings()
                        .one()
                    )
                source = row["authorized_recovery_source_refs"][-1]
                flow = await factory(recovered)
                _, head = await coordinator.load(recovered)

                async def observed(ref):
                    await coordinator.observe_checkpoint(
                        recovered, ref, FLEXIBLE_GRAPH_VERSION
                    )

                final = await flow.resume(
                    head,
                    CheckpointRefV1.model_validate(source["source_ref"]),
                    previous,
                    recovered,
                    request.execution_input(),
                    FLEXIBLE_GRAPH_VERSION,
                    evidence=ExecutionStopEvidenceV1.model_validate(
                        source["stop_evidence"]
                    ),
                    checkpoint_sink=observed,
                )
                await coordinator.stage(recovered, final)
                await coordinator.stopped(
                    recovered,
                    ExecutionStopEvidenceV1(
                        kind="EXECUTOR_STOP_ACK",
                        execution_id=recovered.execution_id,
                        worker_instance_id=recovered.worker_instance_id,
                        generation=recovered.generation,
                        stopped_at=datetime.now(UTC),
                        checkpoint_writes_drained=True,
                        proof_ref="fixture:joined-resume",
                    ),
                    ExecutionOutcome(status="COMPLETED"),
                )
                assert sum(products.calls for products in product_instances) == 1
                async with engine.connect() as conn:
                    resumed_attempts = set(
                        (
                            await conn.execute(
                                select(agent_call_attempts.c.id).where(
                                    agent_call_attempts.c.run_id == submitted.run_id
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                assert original_attempts < resumed_attempts
                view = await coordinator.get(owner, conversation.id, submitted.run_id)
                expected = "COMPLETED"
                record = await store.get(owner, conversation.id)
            if tamper:
                assert executor.last_error == "STOP_ACK_NOT_PERSISTED"
                assert record.revision == 0 and record.active_run_id == submitted.run_id
                async with engine.connect() as conn:
                    messages = (
                        (
                            await conn.execute(
                                select(conversation_messages).where(
                                    conversation_messages.c.conversation_id == record.id
                                )
                            )
                        )
                        .mappings()
                        .all()
                    )
                assert [m["role"] for m in messages] == ["user"]
                return
            assert view.status == expected, executor.last_error
            assert record.revision == 1 and record.active_run_id is None
            async with engine.connect() as conn:
                persisted = (
                    (
                        await conn.execute(
                            select(agent_runs).where(
                                agent_runs.c.id == submitted.run_id
                            )
                        )
                    )
                    .mappings()
                    .one()
                )
            assert (
                persisted["last_checkpoint_ref"]["checkpoint_id"]
                == record.accepted_checkpoint_id
            )
            assert persisted["versions"]["graph_version"] == FLEXIBLE_GRAPH_VERSION
            snapshot = await reader.read(
                CheckpointRefV1(
                    thread_id=record.id, checkpoint_id=record.accepted_checkpoint_id
                )
            )
            assert snapshot.flexible.status == expected
            assert snapshot.flexible.task_message_id == request.request_id
            assert (await coordinator.submit(owner, request)).run_id == submitted.run_id
            async with engine.connect() as conn:
                messages = (
                    (
                        await conn.execute(
                            select(conversation_messages)
                            .where(conversation_messages.c.conversation_id == record.id)
                            .order_by(conversation_messages.c.sequence)
                        )
                    )
                    .mappings()
                    .all()
                )
            assert [m["role"] for m in messages] == ["user", "assistant"]
            assert messages[1]["content"] == snapshot.flexible.assistant_message
            assert executor.last_error is None
    finally:
        if executor:
            assert await executor.stop(grace_seconds=5)
        if conversation:
            async with engine.begin() as conn:
                await conn.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
