"""Managed confirm/build/explain with real optimizer, SQL and native checkpoints."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine
from test_flexible_graph_postgres import Budget, Catalog

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.application.ports.hardware import ComponentSpec
from ai_service.application.ports.web_search import WebPageReadResult
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import (
    SavedBuildEvidence,
    ToolContext,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
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
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import (
    agent_call_attempts,
    conversation_messages,
    conversations,
)


def hardware():
    components = [
        ComponentSpec(
            name="CPU",
            category="CPU",
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
        ),
        ComponentSpec(
            name="Mainboard",
            category="MAINBOARD",
            price=3_400_000,
            socket="AM5",
            ram_type="DDR5",
            form_factor="Micro-ATX",
            ram_slots=4,
        ),
        ComponentSpec(
            name="RAM 32GB",
            category="RAM",
            price=2_400_000,
            ram_type="DDR5",
            capacity_gb=32,
            performance_score=78,
        ),
        ComponentSpec(
            name="GPU",
            category="GPU",
            price=7_000_000,
            tdp_watts=115,
            gpu_length_mm=240,
            vram_gb=8,
            performance_score=76,
        ),
        ComponentSpec(
            name="Case",
            category="CASE",
            price=1_100_000,
            supported_form_factors=["Micro-ATX"],
            max_gpu_length_mm=330,
            max_cooler_height_mm=161,
        ),
        ComponentSpec(
            name="Cooler",
            category="COOLER",
            price=500_000,
            cooler_height_mm=140,
            supported_sockets=["AM5"],
            performance_score=75,
        ),
        ComponentSpec(
            name="SSD",
            category="STORAGE",
            price=950_000,
            capacity_gb=500,
            performance_score=65,
        ),
        ComponentSpec(
            name="PSU",
            category="PSU",
            price=1_100_000,
            wattage=650,
            psu_tier="C",
            performance_score=72,
        ),
    ]
    return [
        CatalogRowV1(
            variant_id=(identity := uuid4()),
            sku=f"SKU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=part.model_copy(update={"id": identity}),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["fixture:canonical-hardware"],
        )
        for index, part in enumerate(components)
    ]


class Products(Catalog):
    async def resolve(self, mention):
        return []


class Extractor:
    calls = 0

    def __init__(self, multi=False):
        self.multi = multi

    async def extract_tasks(self, message):
        self.calls += 1
        batch = TaskDraftBatch(
            tasks=[
                {
                    "kind": "EXPLAIN_SAVED_BUILD"
                    if message == "Explain saved build"
                    else "BUILD",
                    "start": 0,
                    "end": len(message),
                    "quote": message,
                }
            ]
        )
        if self.multi and message != "Explain saved build":
            batch.tasks.append(batch.tasks[0].model_copy(update={"kind": "ANSWER"}))
        return batch


class Model:
    def __init__(self, missing_unit=False, missing_reference=False, multi=False):
        self.missing_unit = missing_unit
        self.missing_reference = missing_reference
        self.multi = multi

    async def decide(self, message, transcript, allowed, repair, context=None):
        if message == "Explain saved build":
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "Unverified price",
                        "evidence_refs": [context.saved_build.evidence_ref],
                    },
                }
            )
        if not context.requirements.requirements:
            source = "30000000 VND GAMING_1080P"
            decision = FlexibleDecision(
                action={
                    "kind": "PROPOSE_REQUIREMENTS",
                    "source_text": source,
                    "draft": {
                        "items": [
                            {
                                "subject": "SETUP",
                                "predicate": "budget",
                                "operator": "LTE",
                                "start": 0,
                                "end": len("30000000 VND"),
                                "value": {
                                    "kind": "QUANTITY",
                                    "amount": 30_000_000,
                                    "unit": "VND",
                                },
                            },
                            {
                                "target": "GOAL",
                                "subject": "WORKLOAD",
                                "predicate": "use_case",
                                "start": len("30000000 VND "),
                                "end": len(source),
                                "value": {"kind": "TEXT", "text": "GAMING_1080P"},
                            },
                        ]
                    },
                }
            )
            if self.missing_unit:
                from ai_service.capabilities.assistant.requirements.resolver import (
                    DraftItem,
                )

                decision.action.source_text += " RAM 32"
                decision.action.draft.items.append(
                    DraftItem(
                        subject="RAM",
                        predicate="capacity",
                        operator="GTE",
                        start=len(source) + 1,
                        end=len(source) + len(" RAM 32"),
                        value={"kind": "UNRESOLVED_QUANTITY", "amount": 32},
                    )
                )
            if self.missing_reference:
                from ai_service.capabilities.assistant.requirements.resolver import (
                    DraftItem,
                )

                decision.action.source_text += " RTX 5070 Ti"
                decision.action.draft.items.append(
                    DraftItem(
                        subject="GPU",
                        predicate="identity",
                        start=len(source) + 1,
                        end=len(source) + len(" RTX 5070 Ti"),
                        value={"kind": "TEXT", "text": "RTX 5070 Ti"},
                    )
                )
            return decision
        if not transcript:
            return FlexibleDecision(
                action={
                    "kind": "CALL_TOOL",
                    "call": {"call_id": "build", "name": "build_pc", "arguments": {}},
                }
            )
        if self.multi and len(transcript) == 1:
            return FlexibleDecision(
                action={
                    "kind": "CALL_TOOL",
                    "call": {
                        "call_id": "page",
                        "name": "read_web_page",
                        "arguments": {"url": "https://example.com/specs"},
                    },
                }
            )
        if self.multi:
            build, page = transcript[0].result, transcript[1].result
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "MULTI_TASK",
                    "answer": {"text": "Ignored aggregate"},
                    "task_answers": [
                        {
                            "task_index": 0,
                            "outcome": "BUILT",
                            "answer": {
                                "text": "Every game 4K",
                                "build_ref": build.result.build_ref,
                                "evidence_refs": build.evidence_refs,
                            },
                        },
                        {
                            "task_index": 1,
                            "outcome": "INSUFFICIENT_EVIDENCE",
                            "supporting_tool_call_ids": [page.tool_call_id],
                            "answer": {
                                "text": "999 mm",
                                "evidence_refs": [page.result.facts[0].evidence_ref],
                            },
                        },
                    ],
                }
            )
        outcome = transcript[-1].result
        return FlexibleDecision(
            action={
                "kind": "FINAL_ANSWER",
                "outcome": "BUILT",
                "answer": {
                    "text": "Unverified text",
                    "build_ref": outcome.result.build_ref,
                    "evidence_refs": outcome.evidence_refs,
                },
            }
        )


class CountingService(PCBuildApplicationService):
    calls = 0

    def build_pc(self, *args, **kwargs):
        self.calls += 1
        return super().build_pc(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [
        "NONE",
        "UNIT",
        "REFERENCE",
        "UNIT_FREE",
        "REFERENCE_FREE",
        "UNIT_AMEND",
        "UNIT_DISCARD",
        "MULTI",
    ],
)
async def test_managed_confirmation_build_and_saved_explanation_preserve_snapshot(
    postgres_dsn,
    selection,
):
    missing_unit = selection.startswith("UNIT")
    missing_reference = selection.startswith("REFERENCE")
    amend_choice = selection in {"UNIT_AMEND", "UNIT_DISCARD"}
    free_choice = selection.endswith("FREE") or amend_choice
    needs_selection = missing_unit or missing_reference
    multi = selection == "MULTI"
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
    service, extractor, catalog = CountingService(), Extractor(multi), hardware()

    class Reader:
        calls = 0

        async def read(self, url):
            self.calls += 1
            return WebPageReadResult(
                url=url,
                final_url=url,
                status="READABLE",
                extraction_status="STRUCTURED",
                document_hash="a" * 64,
                facts=[
                    {
                        "model": "GPU",
                        "value": None,
                        "fact_status": "UNKNOWN",
                        "source_url": url,
                        "document_hash": "a" * 64,
                        "parser_version": "specs-v1",
                        "locator": "dimensions",
                        "quote": "Dimensions not provided",
                    }
                ],
            )

    web_reader = Reader()

    class Choices:
        async def choose_unit(self, message, options):
            from ai_service.capabilities.assistant.requirements.quantity_clarification import (
                UnitChoiceDraft,
            )

            if amend_choice:
                return UnitChoiceDraft()
            return UnitChoiceDraft(
                selections=[
                    {"item_index": 2, "unit": "GB", "start": 0, "end": 2, "quote": "GB"}
                ]
            )

        async def choose(self, message, options):
            from ai_service.capabilities.assistant.requirements.resolver import (
                ChoiceDraft,
            )

            return ChoiceDraft(
                selections=[
                    {
                        "item_index": 2,
                        "variant_id": gpu.variant_id,
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

    choices = Choices()

    class Amender:
        async def amend(self, message, pending):
            from ai_service.capabilities.assistant.requirements.amendment import (
                PendingAmendment,
            )

            if selection == "UNIT_DISCARD":
                return PendingAmendment(
                    discards=[
                        {
                            "item_index": 2,
                            "start": 0,
                            "end": len(message),
                            "quote": message,
                        }
                    ]
                )
            return PendingAmendment(
                replacements=[
                    {
                        "item_index": 2,
                        "item": {
                            "subject": "RAM",
                            "predicate": "capacity",
                            "operator": "GTE",
                            "start": 0,
                            "end": len(message),
                            "value": {"kind": "QUANTITY", "amount": 64, "unit": "GB"},
                        },
                    }
                ]
            )

    if selection == "UNIT_AMEND":
        ram = next(row for row in catalog if row.component.category == "RAM")
        ram64_id = uuid4()
        catalog.append(
            ram.model_copy(
                update={
                    "variant_id": ram64_id,
                    "sku": "RAM64",
                    "component": ram.component.model_copy(
                        update={"id": ram64_id, "capacity_gb": 64, "name": "RAM 64GB"}
                    ),
                },
                deep=True,
            )
        )
    if missing_reference:
        gpu = next(row for row in catalog if row.component.category == "GPU")
        gpu.component.name = "RTX 5070 Ti A"
        other_id = uuid4()
        catalog.append(
            gpu.model_copy(
                update={
                    "variant_id": other_id,
                    "sku": "GPU-OTHER",
                    "component": gpu.component.model_copy(
                        update={"id": other_id, "name": "RTX 5070 Ti B"}
                    ),
                },
                deep=True,
            )
        )
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            conversation = await store.create(owner, "Managed build fixture")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(
                engine, saver, request_versions=(2,), flexible_service=service
            )
            reader = PostgresAcceptedCheckpointReader(saver)

            async def factory(claim):
                request, head = await coordinator.load(claim)
                accepted = await reader.read(head.accepted_checkpoint_ref)
                ctx = ToolContext(
                    account_id=await coordinator.owner(claim),
                    conversation_id=claim.conversation_id,
                    run_id=claim.run_id,
                    requirements=accepted.requirement_state,
                    catalog=catalog,
                    read_catalog_allowed=True,
                    execution_allowed=True,
                    saved_build=SavedBuildEvidence(
                        source_ref=head.accepted_checkpoint_ref,
                        snapshot=accepted.public_build,
                    )
                    if accepted.public_build
                    else None,
                )
                pending = (
                    accepted.flexible
                    if accepted.flexible and accepted.flexible.status == "WAITING_INPUT"
                    else None
                )
                required = (
                    [task.task for task in pending.task_outcomes]
                    if pending and not pending.pending_task_resolution
                    else []
                )
                products = Products()
                graph = build_flexible_graph(
                    Model(missing_unit, missing_reference, multi),
                    ToolDispatcher(
                        products, service, reader=web_reader if multi else None
                    ),
                    ctx,
                    required,
                    AttemptBoundary(coordinator.ledger, claim) if multi else Budget(),
                    conversation_revision=head.revision,
                    publication=FlexiblePublicationInput.for_request(accepted, request),
                    task_extractor=None if required else extractor,
                    task_catalog=None if required else products,
                    checkpointer=saver,
                    clarifier=choices if free_choice else None,
                    unit_clarifier=choices if free_choice else None,
                    amender=Amender() if amend_choice else None,
                )
                return DurableConversationGraph(
                    graph, accepted_versions={FLEXIBLE_GRAPH_VERSION}
                )

            executor = ManagedExecutor(
                coordinator,
                ConversationExecution(coordinator, factory, FLEXIBLE_GRAPH_VERSION),
                worker_id=uuid4(),
                poll_seconds=0.02,
            )
            await executor.start()

            async def run(request, expected):
                submitted = await coordinator.submit(owner, request)
                executor.wake()
                for _ in range(150):
                    view = await coordinator.get(
                        owner, conversation.id, submitted.run_id
                    )
                    if view.status == expected or executor.last_error:
                        break
                    await asyncio.sleep(0.02)
                assert view.status == expected, executor.last_error
                record = await store.get(owner, conversation.id)
                return await reader.read(
                    CheckpointRefV1(
                        thread_id=record.id, checkpoint_id=record.accepted_checkpoint_id
                    )
                )

            first = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="Build PC 30 triệu gaming 1080p; hỏi kích thước GPU"
                if multi
                else "Build PC 30 triệu gaming 1080p",
            )
            waiting = await run(first, "WAITING_INPUT")
            assert not waiting.requirement_state.requirements
            assert service.calls == 0
            if needs_selection:
                pending = waiting.flexible.pending_resolution
                assert pending is not None
                assert waiting.flexible.resolution_options is not None
                command = (
                    {
                        "unit_reply": {
                            "pending_message_id": pending.draft.message_id,
                            "constraints_revision": pending.draft.constraints_revision,
                            "selections": [
                                {
                                    "item_index": 2,
                                    "unit": "GB",
                                    "start": 0,
                                    "end": 2,
                                    "quote": "GB",
                                }
                            ],
                        }
                    }
                    if missing_unit
                    else {
                        "reference_reply": {
                            "pending_message_id": pending.draft.message_id,
                            "selections": [
                                {"item_index": 2, "variant_id": gpu.variant_id}
                            ],
                        }
                    }
                )
                if free_choice:
                    command = {}
                waiting = await run(
                    TurnRequestV2(
                        conversation_id=conversation.id,
                        request_id=uuid4(),
                        expected_revision=1,
                        message="64GB"
                        if selection == "UNIT_AMEND"
                        else "Bỏ RAM"
                        if selection == "UNIT_DISCARD"
                        else "GB"
                        if missing_unit
                        else "Choose A",
                        **command,
                    ),
                    "WAITING_INPUT",
                )
                assert waiting.flexible.pending_resolution is None
                assert waiting.flexible.task_message_id == first.request_id
                assert not waiting.requirement_state.requirements
                assert service.calls == 0 and extractor.calls == 1
            proposal = waiting.flexible.pending_proposal
            second = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=2 if needs_selection else 1,
                message="OK",
                confirmation={
                    "proposal_id": proposal.proposal_id,
                    "proposal_hash": proposal.proposal_hash,
                    "target_revision": proposal.target_revision,
                    "base_requirements_hash": proposal.base_requirements_hash,
                },
            )
            built = await run(second, "COMPLETED")
            assert built.flexible.answer.outcome == ("MULTI_TASK" if multi else "BUILT")
            if multi:
                assert built.flexible.status == "UNRESOLVED"
                assert [task.status for task in built.flexible.task_outcomes] == [
                    "COMPLETED",
                    "UNRESOLVED",
                ]
                assert built.flexible.task_outcomes[1].reason == "FACT_UNKNOWN"
                assert (
                    "999" not in built.flexible.assistant_message
                    and "Every game" not in built.flexible.assistant_message
                )
                assert web_reader.calls == 1
                before = await store.get(owner, conversation.id)
                repeated = await coordinator.submit(owner, second)
                accepted_record = await saver.aget_tuple(
                    DurableConversationGraph.config(
                        CheckpointRefV1(
                            thread_id=conversation.id,
                            checkpoint_id=before.accepted_checkpoint_id,
                        )
                    )
                )
                assert str(repeated.run_id) == accepted_record.metadata["app_run_id"]
                assert repeated.status == "COMPLETED"
                async with engine.connect() as conn:
                    attempts = (
                        (
                            await conn.execute(
                                select(agent_call_attempts).where(
                                    agent_call_attempts.c.run_id == repeated.run_id
                                )
                            )
                        )
                        .mappings()
                        .all()
                    )
                assert sum(row["kind"] == "TOOL_IO" for row in attempts) == 2
                assert any(row["kind"] == "MODEL" for row in attempts)
                assert all(row["status"] == "SUCCEEDED" for row in attempts)
                after = await store.get(owner, conversation.id)
                assert after.revision == before.revision and after.active_run_id is None
            assert built.flexible.task_message_id == first.request_id
            assert built.public_build.total_spending_vnd == sum(
                part.spending_vnd for part in built.public_build.parts
            )
            assert 0 < built.public_build.total_spending_vnd <= 30_000_000
            assert service.calls == 1 and extractor.calls == 1
            if selection == "UNIT_AMEND":
                assert any(
                    part.variant_id == ram64_id for part in built.public_build.parts
                )
                assert (
                    next(
                        req
                        for req in built.requirement_state.requirements.values()
                        if req.subject == "RAM"
                    ).value.amount
                    == 64
                )
            if selection == "UNIT_DISCARD":
                assert not any(
                    req.subject == "RAM"
                    for req in built.requirement_state.requirements.values()
                )
            if missing_reference:
                assert any(
                    part.variant_id == gpu.variant_id
                    for part in built.public_build.parts
                )
            historical = built.public_build.model_dump(mode="json")
            explained = await run(
                TurnRequestV2(
                    conversation_id=conversation.id,
                    request_id=uuid4(),
                    expected_revision=3 if needs_selection else 2,
                    message="Explain saved build",
                ),
                "COMPLETED",
            )
            assert explained.public_build.model_dump(mode="json") == historical
            assert explained.requirement_state == built.requirement_state
            assert (
                explained.flexible.task_outcomes[0].task.kind == "EXPLAIN_SAVED_BUILD"
            )
            assert "Unverified price" not in explained.flexible.assistant_message
            assert service.calls == 1 and extractor.calls == 2
            record = await store.get(owner, conversation.id)
            assert (
                record.revision == (4 if needs_selection else 3)
                and record.active_run_id is None
            )
            assert (await coordinator.submit(owner, second)).status == "COMPLETED"
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
            assert [m["role"] for m in messages] == ["user", "assistant"] * (
                4 if needs_selection else 3
            )
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
