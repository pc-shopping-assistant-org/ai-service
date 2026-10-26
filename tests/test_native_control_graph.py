import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_explicit_user_admission import draft
from test_flexible_publication import base_snapshot
from test_native_react_model import Provider
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.source_quotes import project_requirement_draft
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.providers.react_model import (
    GEMINI_SIGNATURES_KEY,
    NativeFlexibleModel,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_native_control_acknowledgement_precedes_later_tool_pairs(resume):
    trusted, products, provider = context(), Catalog(), Provider(None, name="gemini")
    calls = 0
    extracted = 0
    interrupted = False

    class Extractor:
        async def extract(self, message, requirements):
            nonlocal extracted
            extracted += 1
            return draft()

    async def respond(messages):
        nonlocal calls, interrupted
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "proposal",
                        "name": "propose_requirements",
                        "args": {
                            "draft": project_requirement_draft(
                                draft(), "RAM 32GB"
                            ).model_dump(mode="json"),
                        },
                    }
                ],
                additional_kwargs={GEMINI_SIGNATURES_KEY: {"proposal": "signature"}},
            )
        paired = [message for message in messages if isinstance(message, ToolMessage)]
        assert paired[0].tool_call_id == "proposal"
        assert json.loads(paired[0].content)["code"] == "REQUIREMENTS_ACCEPTED"
        assistant = next(
            message for message in messages if isinstance(message, AIMessage)
        )
        assert assistant.additional_kwargs[GEMINI_SIGNATURES_KEY] == {
            "proposal": "signature"
        }
        if len(paired) == 1:
            if resume and not interrupted:
                interrupted = True
                raise TimeoutError("after control transition checkpoint")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search",
                        "name": "search_catalog",
                        "args": {"query": "RAM"},
                    }
                ],
            )
        assert [message.tool_call_id for message in paired] == ["proposal", "search"]
        refs = json.loads(paired[-1].content)["evidence_refs"]
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "submit",
                    "name": "submit_result",
                    "args": {
                        "results": [
                            {
                                "text": "Lựa chọn",
                                "evidence_refs": refs,
                            }
                        ]
                    },
                }
            ],
        )

    provider.model.ainvoke = respond
    flow = build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(products, PCBuildApplicationService()),
        trusted,
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=0,
        user_extractor=Extractor(),
        checkpointer=InMemorySaver(),
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(trusted),
            task_message_id=uuid4(),
            task_message="RAM 32GB",
            request_message="RAM 32GB",
        ),
    )
    config = {"configurable": {"thread_id": str(trusted.conversation_id)}}
    if resume:
        with pytest.raises(TimeoutError):
            await flow.ainvoke({"message": "RAM 32GB"}, config, durability="sync")
        snapshot = await flow.aget_state(config)
        assert len(snapshot.values["control_history"]) == 1
        terminal = await flow.ainvoke(None, snapshot.config, durability="sync")
    else:
        terminal = await flow.ainvoke(
            {"message": "RAM 32GB"}, config, durability="sync"
        )
    assert calls == (4 if resume else 3) and products.calls == 1 and extracted == 1
    assert len(terminal["control_history"]) == 1
    assert [entry["call"]["name"] for entry in terminal["ledger"]] == ["search_catalog"]
    assert terminal["accepted_snapshot"]["flexible"]["answer"]["status"] == "COMPLETED"
