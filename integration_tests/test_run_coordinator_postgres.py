"""Main run path uses real SQL admission, managed execution and native checkpoints."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import ConversationNotFound
from ai_service.capabilities.assistant.stateful_contracts import TurnRequestV1
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import (
    PostgresRunCoordinator,
    RunAdmissionError,
)
from ai_service.infrastructure.persistence.schema import (
    conversation_messages,
    conversations,
)


@pytest.mark.asyncio
async def test_submit_claim_execute_publish_and_follow_up(postgres_dsn):
    from ai_service.application.ports.hardware import ComponentCategory as C
    from ai_service.application.ports.hardware import ComponentSpec
    from ai_service.capabilities.assistant.provenance import CatalogRowV1
    from ai_service.capabilities.assistant.requirements.contracts import (
        QuantityValue,
        TextValue,
    )
    from ai_service.capabilities.assistant.requirements.resolver import Draft, DraftItem
    from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
    from ai_service.infrastructure.graph.durable import DurableConversationGraph
    from ai_service.infrastructure.graph.requirement_build import (
        build_requirement_graph,
    )
    from ai_service.infrastructure.persistence.accepted_state import (
        PostgresAcceptedCheckpointReader,
    )

    class Extractor:
        async def extract(self, message, state):
            if message == "Lỗi model":
                raise TimeoutError("test provider failure")
            return (
                Draft(items=[])
                if state.requirements
                else Draft(
                    items=[
                        DraftItem(
                            subject="SETUP",
                            predicate="budget",
                            operator="LTE",
                            value=QuantityValue(amount=30_000_000, unit="VND"),
                            start=0,
                            end=len(message),
                        ),
                        DraftItem(
                            target="GOAL",
                            subject="WORKLOAD",
                            predicate="use_case",
                            value=TextValue(text="GAMING_1080P"),
                            start=0,
                            end=len(message),
                        ),
                    ]
                )
            )

    class Catalog:
        async def load(self, draft, state):
            return catalog

    components = [
        ComponentSpec(
            name="CPU",
            category=C.CPU,
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
        ),
        ComponentSpec(
            name="Mainboard",
            category=C.MAINBOARD,
            price=3_400_000,
            socket="AM5",
            ram_type="DDR5",
            form_factor="Micro-ATX",
            ram_slots=4,
        ),
        ComponentSpec(
            name="RAM 32GB",
            category=C.RAM,
            price=2_400_000,
            ram_type="DDR5",
            capacity_gb=32,
            performance_score=78,
        ),
        ComponentSpec(
            name="GPU",
            category=C.GPU,
            price=7_000_000,
            tdp_watts=115,
            gpu_length_mm=240,
            vram_gb=8,
            performance_score=76,
        ),
        ComponentSpec(
            name="Case",
            category=C.CASE,
            price=1_100_000,
            supported_form_factors=["Micro-ATX"],
            max_gpu_length_mm=330,
            max_cooler_height_mm=161,
        ),
        ComponentSpec(
            name="Cooler",
            category=C.COOLER,
            price=500_000,
            cooler_height_mm=140,
            supported_sockets=["AM5"],
            performance_score=75,
        ),
        ComponentSpec(
            name="SSD",
            category=C.STORAGE,
            price=950_000,
            capacity_gb=500,
            performance_score=65,
        ),
        ComponentSpec(
            name="PSU",
            category=C.PSU,
            price=1_100_000,
            wattage=650,
            psu_tier="C",
            performance_score=72,
        ),
    ]
    catalog = [
        CatalogRowV1(
            variant_id=(identity := uuid4()),
            sku=f"SKU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=part.model_copy(update={"id": identity}),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["test/canonical-catalog"],
        )
        for index, part in enumerate(components)
    ]

    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    owner, worker = uuid4(), uuid4()
    store = PostgresConversationStore(engine)
    conversation = await store.create(owner, "Run main flow")
    executor = None
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            graph = DurableConversationGraph(
                build_requirement_graph(Extractor(), Catalog(), checkpointer=saver)
            )
            coordinator = PostgresRunCoordinator(engine, saver)

            execute = ConversationExecution(
                coordinator, lambda claim: graph, "run-core-v1"
            )

            request = TurnRequestV1(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="Build 30 triệu gaming",
            )
            submitted = await coordinator.submit(owner, request)
            assert (await coordinator.submit(owner, request)).run_id == submitted.run_id
            with pytest.raises(RunAdmissionError, match="AI_REQUEST_CONFLICT"):
                await coordinator.submit(
                    owner, request.model_copy(update={"message": "Different body"})
                )
            with pytest.raises(RunAdmissionError, match="AI_CONVERSATION_BUSY"):
                await coordinator.submit(
                    owner, request.model_copy(update={"request_id": uuid4()})
                )
            with pytest.raises(RunAdmissionError, match="AI_CONVERSATION_NOT_FOUND"):
                await coordinator.submit(uuid4(), request)
            executor = ManagedExecutor(
                coordinator, execute, worker_id=worker, poll_seconds=0.01
            )
            await executor.start()

            async def completed(run_id, status="COMPLETED"):
                async with asyncio.timeout(5):
                    while (
                        view := await coordinator.get(owner, conversation.id, run_id)
                    ).status.value != status:
                        await asyncio.sleep(0.01)
                return view

            await completed(submitted.run_id)
            first = await store.get(owner, conversation.id)
            assert first.revision == 1 and first.active_run_id is None
            # Repeating an accepted request returns the old run before revision checks.
            assert (await coordinator.submit(owner, request)).run_id == submitted.run_id
            with pytest.raises(RunAdmissionError, match="AI_REVISION_CONFLICT"):
                await coordinator.submit(
                    owner, request.model_copy(update={"request_id": uuid4()})
                )
            from ai_service.application.use_cases.stateful_conversations import (
                StatefulConversationService,
                StatefulRunService,
            )
            from ai_service.capabilities.assistant.public_contracts import RunViewV1
            from ai_service.config.settings import Settings
            from ai_service.infrastructure.stateful import StatefulResources
            from ai_service.main import create_app

            app = create_app(
                Settings(
                    _env_file=None,
                    stateful_enabled=True,
                    database_url=postgres_dsn.replace(
                        "postgresql://", "postgresql+psycopg://", 1
                    ),
                    jwt_jwks_url="http://identity.test/jwks",
                )
            )

            # JWT boundary has separate tests; this gate uses real SQL/native
            # execution with an injected authenticated owner and runtime graph.
            class Auth:
                async def authenticate(self, token):
                    assert token == "test-owner"
                    return owner

            app.state.stateful = StatefulResources(
                StatefulConversationService(
                    store,
                    PostgresAcceptedCheckpointReader(saver),
                    NativeConversationInitializer(saver),
                ),
                Auth(),
                runs=StatefulRunService(coordinator, executor.wake),
            )
            next_request = request.model_copy(
                update={
                    "request_id": uuid4(),
                    "expected_revision": 1,
                    "message": "Giữ nguyên",
                }
            )
            path = f"/api/v1/conversations/{conversation.id}/runs"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test-owner"},
            ) as client:
                response = await client.post(
                    path, json=next_request.model_dump(mode="json")
                )
                assert response.status_code == 202
                follow_up = RunViewV1.model_validate(response.json()["data"])
            # Closing the HTTP client does not cancel the managed execution.
            await completed(follow_up.run_id)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test-owner"},
            ) as client:
                status = await client.get(path + f"/{follow_up.run_id}")
                assert (
                    status.status_code == 200
                    and status.json()["data"]["status"] == "COMPLETED"
                )
                replay = await client.post(
                    path, json=next_request.model_dump(mode="json")
                )
                assert replay.status_code == 202 and replay.json()["data"][
                    "run_id"
                ] == str(follow_up.run_id)
            second = await store.get(owner, conversation.id)
            assert (
                second.revision == 2
                and second.accepted_checkpoint_id != first.accepted_checkpoint_id
            )
            async with engine.connect() as conn:
                history = (
                    (
                        await conn.execute(
                            select(conversation_messages)
                            .where(
                                conversation_messages.c.conversation_id
                                == conversation.id
                            )
                            .order_by(conversation_messages.c.sequence)
                        )
                    )
                    .mappings()
                    .all()
                )
            assert [row["role"] for row in history] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert "Tổng chi phí linh kiện" in history[-1]["content"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test-owner"},
            ) as client:
                message_path = f"/api/v1/conversations/{conversation.id}/messages"
                first_page = await client.get(message_path, params={"limit": 2})
                assert first_page.status_code == 200
                assert first_page.json()["message"] == "AI_MESSAGES_LISTED"
                page = first_page.json()["data"]
                assert [m["sequence"] for m in page["items"]] == [1, 2]
                assert page["next_cursor"] is not None
                next_page = await client.get(
                    message_path, params={"limit": 2, "cursor": page["next_cursor"]}
                )
                assert [m["sequence"] for m in next_page.json()["data"]["items"]] == [
                    3,
                    4,
                ]
                assert next_page.json()["data"]["next_cursor"] is None
                assert (
                    next_page.json()["data"]["items"][-1]["content"]
                    == history[-1]["content"]
                )
                assert (
                    await client.get(message_path, params={"cursor": str(uuid4())})
                ).status_code == 400
                assert (
                    await client.get(message_path, params={"limit": 0})
                ).status_code == 422
                other = await store.create(owner, "Other conversation")
                try:
                    foreign_cursor = await client.get(
                        f"/api/v1/conversations/{other.id}/messages",
                        params={"cursor": page["next_cursor"]},
                    )
                    assert foreign_cursor.status_code == 400
                    assert foreign_cursor.json()["errors"][0]["field"] == "cursor"
                    with pytest.raises(ConversationNotFound):
                        await store.messages(uuid4(), conversation.id, 20, None)
                finally:
                    async with engine.begin() as conn:
                        await conn.execute(
                            delete(conversations).where(conversations.c.id == other.id)
                        )
            accepted = await PostgresAcceptedCheckpointReader(saver).read(
                CheckpointRefV1(
                    thread_id=conversation.id,
                    checkpoint_id=second.accepted_checkpoint_id,
                )
            )
            assert (
                accepted.consultation.constraints.target_budget_vnd.value == 30_000_000
            )
            assert (
                accepted.public_build is not None
                and accepted.public_build.total_spending_vnd <= 30_000_000
            )
            failed = await coordinator.submit(
                owner,
                request.model_copy(
                    update={
                        "request_id": uuid4(),
                        "expected_revision": 2,
                        "message": "Lỗi model",
                    }
                ),
            )
            executor.wake()
            await completed(failed.run_id, "FAILED_TERMINAL")
            after_failure = await store.get(owner, conversation.id)
            assert after_failure.revision == 2
            assert after_failure.accepted_checkpoint_id == second.accepted_checkpoint_id
            assert after_failure.active_run_id is None
    finally:
        if executor is not None:
            assert await executor.stop(grace_seconds=5)
        async with engine.begin() as conn:
            await conn.execute(
                delete(conversations).where(conversations.c.id == conversation.id)
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
