"""100-turn native state/storage gate; no managed admission or live-provider claim."""

import json
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from test_flexible_graph_postgres import Budget, Catalog, Model

from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
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


class Extractor:
    async def extract_tasks(self, message):
        return TaskDraftBatch(
            tasks=[
                {"kind": "ANSWER", "start": 0, "end": len(message), "quote": message}
            ]
        )


@pytest.mark.asyncio
async def test_hundred_native_turns_bound_working_state_and_preserve_old_evidence(
    postgres_dsn,
):
    conversation, owner = uuid4(), uuid4()
    catalog, service, sizes, refs = Catalog(), PCBuildApplicationService(), [], []
    catalog.product.specifications = {"description": "x" * 6000}
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            root = await NativeConversationInitializer(saver).initialize(conversation)
            head = ConversationHeadV1(
                conversation_id=conversation, revision=0, accepted_checkpoint_ref=root
            )
            reader = PostgresAcceptedCheckpointReader(saver)
            for turn in range(100):
                accepted = await reader.read(head.accepted_checkpoint_ref)
                run_id = uuid4()
                ctx = ToolContext(
                    account_id=owner,
                    conversation_id=conversation,
                    run_id=run_id,
                    requirements=accepted.requirement_state,
                    read_catalog_allowed=True,
                    execution_allowed=True,
                )
                request = TurnRequestV2(
                    conversation_id=conversation,
                    request_id=uuid4(),
                    expected_revision=turn,
                    message=f"GPU turn {turn}",
                )
                graph = build_flexible_graph(
                    Model(),
                    ToolDispatcher(catalog, service),
                    ctx,
                    [],
                    Budget(),
                    conversation_revision=turn,
                    publication=FlexiblePublicationInput.for_request(accepted, request),
                    task_extractor=Extractor(),
                    task_catalog=catalog,
                    checkpointer=saver,
                )
                flow = DurableConversationGraph(
                    graph, accepted_versions={FLEXIBLE_GRAPH_VERSION}
                )
                try:
                    ref = await flow.new_turn(
                        head, run_id, request.execution_input(), FLEXIBLE_GRAPH_VERSION
                    )
                except ValueError as exc:
                    raise AssertionError(
                        f"turn={turn + 1} last_state_bytes={sizes[-1] if sizes else 0}: {exc}"
                    ) from exc
                snapshot = await flow.read(ref)
                sizes.append(
                    len(json.dumps(snapshot.values, ensure_ascii=False).encode())
                )
                refs.append(ref)
                head = head.model_copy(
                    update={"revision": turn + 1, "accepted_checkpoint_ref": ref}
                )
            assert catalog.calls == 100
            assert len(snapshot.values["ledger"]) == 1
            assert max(sizes) < 64_000
            first = await flow.read(refs[0])
            assert len(first.values["ledger"]) == 1
            assert (
                first.values["ledger"][0]["result"]["result"]["products"][0][
                    "specifications"
                ]["description"]
                == "x" * 6000
            )
            assert first.metadata["app_run_id"] != snapshot.metadata["app_run_id"]
            cursor = await saver.conn.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS native_payload_bytes FROM ("
                "SELECT SUM(octet_length(checkpoint::text) + octet_length(metadata::text)) AS bytes FROM checkpoints WHERE thread_id = %s "
                "UNION ALL SELECT SUM(octet_length(blob)) FROM checkpoint_blobs WHERE thread_id = %s "
                "UNION ALL SELECT SUM(octet_length(blob)) FROM checkpoint_writes WHERE thread_id = %s"
                ") AS stored",
                (str(conversation),) * 3,
            )
            storage = await cursor.fetchone()
            assert storage["native_payload_bytes"] < 16_000_000
            print(
                f"turns=100 state_highwater_bytes={max(sizes)} native_payload_bytes={storage['native_payload_bytes']}"
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            first = await saver.aget_tuple(DurableConversationGraph.config(refs[0]))
            last = await saver.aget_tuple(DurableConversationGraph.config(refs[-1]))
            assert first is not None and last is not None
            assert (
                len(first.checkpoint["channel_values"]["ledger"])
                == len(last.checkpoint["channel_values"]["ledger"])
                == 1
            )
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation))
