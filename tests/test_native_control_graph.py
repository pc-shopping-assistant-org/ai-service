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
from ai_service.capabilities.assistant.react.model_contracts import RequirementInput
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


def test_gemini_integer_text_with_explicit_unit_is_normalized_before_domain_validation():
    wire = RequirementInput(
        subject="SETUP",
        predicate="budget",
        operator="LTE",
        target="GOAL",
        value="25000000",
        unit="VND",
        quote="25 triệu",
    )
    assert wire.quoted().value.amount == 25000000
    assert (
        RequirementInput(subject="GPU", predicate="model", value="4070", quote="4070")
        .quoted()
        .value.text
        == "4070"
    )
    for invalid in ("25.5", "NaN", "25 triệu", "-1"):
        with pytest.raises(ValueError):
            RequirementInput(
                subject="SETUP",
                predicate="budget",
                value=invalid,
                unit="VND",
                quote="budget",
            ).quoted()


def test_build_tool_does_not_advertise_an_objective_rejected_by_application():
    from ai_service.capabilities.assistant.react.contracts import BuildPCArgs
    from ai_service.infrastructure.providers.tool_schema import model_tool_schema

    assert model_tool_schema(BuildPCArgs)["properties"] == {}


def test_existing_goal_id_determines_update_scope_even_when_model_uses_default_target():
    from test_requirement_pipeline import state

    from ai_service.capabilities.assistant.react.contracts import ToolCall
    from ai_service.capabilities.assistant.react.control_actions import (
        normalize_control,
    )
    from ai_service.capabilities.assistant.requirements.user_admission import (
        admit_source_draft,
        apply_user_receipt,
    )

    trusted = context(state())
    budget = next(
        r for r in trusted.requirements.requirements.values() if r.predicate == "budget"
    )
    trusted.requirements.requirements.pop(budget.id)
    trusted.requirements.goals[budget.id] = budget
    call = ToolCall(
        call_id="update",
        name="propose_requirements",
        arguments={
            "items": [
                {
                    "subject": "SETUP",
                    "predicate": "budget",
                    "target_id": str(budget.id),
                    "operator": "LTE",
                    "value": "25000000",
                    "unit": "VND",
                    "quote": "25 triệu",
                }
            ]
        },
    )
    action = normalize_control(call, [], "25 triệu", trusted).action
    receipt = admit_source_draft(
        trusted.requirements,
        action,
        run_id=trusted.run_id,
        message="25 triệu",
        catalog=[],
    )
    updated = apply_user_receipt(
        trusted.requirements,
        receipt,
        run_id=trusted.run_id,
        message="25 triệu",
        catalog=[],
    )
    assert updated.goals[budget.id].value.amount == 25000000
    assert budget.id not in updated.requirements


@pytest.mark.asyncio
async def test_started_build_requires_canonical_submission_instead_of_plain_fallback(
    realistic_catalog,
):
    from test_requirement_pipeline import rows, state

    trusted = context(state(), rows(realistic_catalog))
    provider = Provider(None, name="gemini")
    calls = 0

    async def respond(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="", tool_calls=[{"id": "build", "name": "build_pc", "args": {}}]
            )
        if calls == 2:
            return AIMessage(content="Cấu hình giá 1 VND, chơi mọi game.")
        assert provider.model.options["tool_choice"] == "submit_result"
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "publish",
                    "name": "submit_result",
                    "args": {
                        "results": [{"text": "Untrusted prose", "calls": ["build"]}]
                    },
                }
            ],
        )

    provider.model.ainvoke = respond
    graph = build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        trusted,
        [],
        Budget(),
        conversation_revision=0,
        task_catalog=Catalog(),
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(trusted),
            task_message_id=uuid4(),
            task_message="Build PC",
            request_message="Build PC",
        ),
    )
    terminal = await graph.ainvoke({"message": "Build PC"})
    assert calls == 3
    assert terminal["answer"]["outcome"] == "BUILT"
    assert "giá 1 VND" not in terminal["answer"]["text"]
    assert terminal["accepted_snapshot"]["public_build"] is not None


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
                            "items": [
                                RequirementInput.from_quoted(item).model_dump(
                                    mode="json", exclude_defaults=True
                                )
                                for item in project_requirement_draft(
                                    draft(), "RAM 32GB"
                                ).items
                            ],
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
                                "refs": refs,
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
    assert calls == (4 if resume else 3) and products.calls == 1 and extracted == 0
    assert len(terminal["control_history"]) == 1
    assert [entry["call"]["name"] for entry in terminal["ledger"]] == ["search_catalog"]
    assert terminal["accepted_snapshot"]["flexible"]["answer"]["status"] == "COMPLETED"
