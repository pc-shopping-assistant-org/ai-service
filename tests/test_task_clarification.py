from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_react_graph import Budget
from test_react_tools import context
from test_task_resolution import Catalog, compare_task, product

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.react.contracts import AnswerCandidate, ToolCall
from ai_service.capabilities.assistant.react.task_resolution import (
    PendingTaskDraft,
    public_task_options,
    resolve_tasks,
    selections_for_pending_task,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.stateful_contracts import (
    TaskClarificationReplyV1,
    TaskProductSelectionV1,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.react import ReactDecision
from ai_service.infrastructure.graph.requirement_build import (
    ReactComposition,
    build_requirement_graph,
)


@pytest.mark.asyncio
async def test_task_choice_budget_denial_cannot_dispatch_model():
    from test_native_requirement_extractor import Budget as ChoiceBudget
    from test_native_requirement_extractor import Provider

    from ai_service.infrastructure.providers.requirement_extractor import (
        NativeRequirementExtractor,
    )

    first, alternative, second = (
        product("ASUS RTX 4060"),
        product("MSI RTX 4060"),
        product("RTX 4070"),
    )
    resolution = await resolve_tasks(
        [compare_task()],
        compare_task().quote,
        CanonicalRequirementState(),
        Catalog({"RTX 4060": [first, alternative], "RTX 4070": [second]}),
    )
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=compare_task().quote,
        constraints_revision=0,
        tasks=[compare_task()],
        resolution=resolution,
    )
    provider = Provider({"selections": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeRequirementExtractor(
            provider, ChoiceBudget(denied=True)
        ).choose_task("Chọn MSI", public_task_options(pending))
    assert provider.model.messages == []


@pytest.mark.asyncio
async def test_offered_product_choice_resolves_original_comparison_and_keeps_unique_binding():
    first, alternative, second = (
        product("ASUS RTX 4060"),
        product("MSI RTX 4060"),
        product("RTX 4070"),
    )
    catalog = Catalog({"RTX 4060": [first, alternative], "RTX 4070": [second]})
    state = CanonicalRequirementState(revision=3)
    initial = await resolve_tasks(
        [compare_task()], compare_task().quote, state, catalog
    )
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=compare_task().quote,
        constraints_revision=3,
        tasks=[compare_task()],
        resolution=initial,
    )
    projection = public_task_options(pending)
    AcceptedConversationSnapshot(
        consultation=ConsultationStateV1(),
        pending_tasks=pending,
        public_task_clarification=projection,
        requirement_state=state,
    )
    with pytest.raises(ValueError, match="revision binding"):
        AcceptedConversationSnapshot(
            consultation=ConsultationStateV1(),
            pending_tasks=pending,
            public_task_clarification=projection,
            requirement_state=CanonicalRequirementState(revision=2),
        )
    with pytest.raises(ValueError, match="projection binding"):
        AcceptedConversationSnapshot(
            consultation=ConsultationStateV1(),
            pending_tasks=pending,
            public_task_clarification=None,
            requirement_state=state,
        )
    reply = TaskClarificationReplyV1(
        pending_message_id=pending.message_id,
        constraints_revision=3,
        selections=[
            TaskProductSelectionV1(
                task_index=0, subject_index=0, product_id=alternative.product_id
            )
        ],
    )
    selected = selections_for_pending_task(pending, reply, state)
    assert any(item.product_id == second.product_id for item in selected)
    resolved = await resolve_tasks(
        pending.tasks, pending.message, state, catalog, selections=selected
    )
    assert not resolved.pending
    assert resolved.requirements[0].product_ids == [
        alternative.product_id,
        second.product_id,
    ]

    # Never substitute another SKU/product when a bound identity disappears.
    catalog.matches["RTX 4060"] = [first]
    unavailable = await resolve_tasks(
        pending.tasks, pending.message, state, catalog, selections=selected
    )
    assert unavailable.requirements == []
    assert unavailable.pending[0].reason == "SELECTED_PRODUCT_UNAVAILABLE"
    with pytest.raises(ValueError, match="candidate"):
        selections_for_pending_task(
            pending,
            reply.model_copy(
                update={
                    "selections": [
                        TaskProductSelectionV1(
                            task_index=0, subject_index=0, product_id=uuid4()
                        )
                    ]
                }
            ),
            state,
        )
    with pytest.raises(ValueError, match="source/revision"):
        selections_for_pending_task(
            pending, reply.model_copy(update={"constraints_revision": 2}), state
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice", ["typed", "free_text", "resume", "empty", "unknown_id", "forged_span"]
)
async def test_comparison_clarification_survives_native_turn_without_reextracting_tasks(
    choice,
):
    from test_native_requirement_extractor import Budget as ChoiceBudget
    from test_native_requirement_extractor import Provider

    from ai_service.infrastructure.providers.requirement_extractor import (
        NativeRequirementExtractor,
    )

    first, alternative, second = (
        product("ASUS RTX 4060"),
        product("MSI RTX 4060"),
        product("RTX 4070"),
    )

    class Products(Catalog):
        lookups = 0

        async def resolve(self, mention):
            self.lookups += 1
            if choice == "resume" and self.lookups == 3:
                raise RuntimeError("Catalog temporarily unavailable after choice")
            return await super().resolve(mention)

        async def get(self, identifier):
            return next(
                (p for p in [first, alternative, second] if p.product_id == identifier),
                None,
            )

    class Extractor:
        calls = 0

        async def extract(self, message, state):
            self.calls += 1
            assert self.calls == 1, (
                "A choice must continue the original task, not create an ANSWER task"
            )
            return Draft(items=[], tasks=[compare_task()])

    class Hardware:
        async def load(self, *args):
            pytest.fail("Comparison must not scan hardware")

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            assert context.task_message == compare_task().quote
            assert context.completion_requirements[0].kind == "COMPARE"
            identifiers = context.completion_requirements[0].product_ids
            if not transcript:
                return ReactDecision(
                    calls=[
                        ToolCall(
                            call_id="compare",
                            name="compare_products",
                            arguments={
                                "product_ids": [
                                    str(identifier) for identifier in identifiers
                                ]
                            },
                        )
                    ]
                )
            return ReactDecision(
                answer=AnswerCandidate(
                    text="Comparison", evidence_refs=transcript[-1].result.evidence_refs
                )
            )

    products = Products({"RTX 4060": [first, alternative], "RTX 4070": [second]})
    extractor = Extractor()
    choice_message = "Chọn MSI"
    output = {
        "selections": []
        if choice == "empty"
        else [
            {
                "task_index": 0,
                "subject_index": 0,
                "product_id": str(
                    uuid4() if choice == "unknown_id" else alternative.product_id
                ),
                "evidence": {
                    "start": 0,
                    "end": len(choice_message),
                    "quote": "Chọn ASUS" if choice == "forged_span" else choice_message,
                },
            }
        ]
    }
    choice_budget = ChoiceBudget()
    native_chooser = NativeRequirementExtractor(Provider(output), choice_budget)
    graph = build_requirement_graph(
        extractor,
        Hardware(),
        checkpointer=InMemorySaver(),
        react=ReactComposition(
            Model(),
            ToolDispatcher(products, PCBuildApplicationService()),
            context(),
            products,
            Budget(),
            task_clarifier=native_chooser if choice != "typed" else None,
        ),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    source = uuid4()
    initial_state = CanonicalRequirementState(revision=3)
    initial = await graph.ainvoke(
        {
            "message": compare_task().quote,
            "message_id": str(source),
            "requirement_state": initial_state.model_dump(mode="json"),
        },
        config,
    )
    assert initial["outcome"] == "CLARIFICATION"
    assert initial["public_task_clarification"]["items"][0]["options"][1][
        "product_id"
    ] == str(alternative.product_id)

    # No explicit selection: retain the task rather than guessing an identity.
    if choice == "typed":
        still_pending = await graph.ainvoke(
            {"message": "Tôi cần chọn bản nào?", "message_id": str(uuid4())}, config
        )
        assert still_pending["outcome"] == "CLARIFICATION"
        assert still_pending["pending_tasks"]["message_id"] == str(source)

    reply = TaskClarificationReplyV1(
        pending_message_id=source,
        constraints_revision=3,
        selections=[
            TaskProductSelectionV1(
                task_index=0, subject_index=0, product_id=alternative.product_id
            )
        ],
    )
    inputs = {
        "message": choice_message,
        "message_id": str(uuid4()),
    }
    if choice == "typed":
        inputs["task_clarification_reply"] = reply.model_dump(mode="json")
    if choice in {"unknown_id", "forged_span"}:
        with pytest.raises(ValueError, match="candidate|evidence"):
            await graph.ainvoke(inputs, config)
        assert len(choice_budget.reservations) == 1
        return
    if choice == "resume":
        with pytest.raises(RuntimeError, match="Catalog temporarily"):
            await graph.ainvoke(inputs, config)
        assert (await graph.aget_state(config)).next == ("resolve_completion",)
        final = await graph.ainvoke(None, config)
    else:
        final = await graph.ainvoke(inputs, config)
    if choice == "empty":
        assert final["outcome"] == "CLARIFICATION"
        assert final["requirement_state"] == initial_state.model_dump(mode="json")
        assert final["pending_tasks"]["message_id"] == str(source)
        return
    assert final["outcome"] == "ANSWERED"
    assert final["task_resolution"]["requirements"][0]["kind"] == "COMPARE"
    assert alternative.name in final["answer"] and second.name in final["answer"]
    assert first.name not in final["answer"]
    assert final["requirement_state"] == initial_state.model_dump(mode="json")
    assert final["pending_tasks"] is None and final["public_task_clarification"] is None
    assert final["task_clarification_reply"] is None
    assert len(choice_budget.reservations) == int(choice in {"free_text", "resume"})
