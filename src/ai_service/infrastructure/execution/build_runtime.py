"""Managed V2 composition; one graph and one executor lifecycle."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_service.application.call_attempts import AttemptBoundary, ExecutionClaim
from ai_service.application.use_cases.stateful_conversations import StatefulRunService
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    SavedBuildEvidence,
    ToolContext,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexibleExecutionContext,
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.catalog_policy import (
    draft_needs_catalog,
)
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.web_search.source_policy import ResearchSourcePolicy
from ai_service.config.settings import Settings
from ai_service.infrastructure.commerce.hardware_catalog import (
    BackendRequirementCatalog,
    HardwareCatalogMapping,
)
from ai_service.infrastructure.commerce.product_catalog import BackendProductCatalog
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.flexible import (
    FLEXIBLE_GRAPH_VERSION,
    build_flexible_graph,
)
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.user_memory import PostgresUserMemoryStore
from ai_service.infrastructure.providers.factory import build_model_provider
from ai_service.infrastructure.providers.react_model import (
    NativeFlexibleModel,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)
from ai_service.infrastructure.providers.task_extractor import NativeTaskExtractor
from ai_service.infrastructure.search.budgeted import (
    BudgetedWebReader,
    BudgetedWebSearch,
)
from ai_service.infrastructure.search.duckduckgo_adapter import DuckDuckGoSearchAdapter
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader

from .conversation import ConversationExecution
from .supervisor import ManagedExecutor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_build_runtime(
    settings: Settings,
    engine: AsyncEngine,
    saver: BaseCheckpointSaver,
    client: httpx.AsyncClient,
):
    mapping = HardwareCatalogMapping.model_validate(settings.hardware_catalog_mapping or {})
    provider = build_model_provider(settings)
    if provider is None:
        raise ValueError("Stateful execution requires a model provider")
    service = PCBuildApplicationService()
    coordinator = PostgresRunCoordinator(
        engine,
        saver,
        request_versions=(2,),
        flexible_service=service,
        max_attempts=2,
        auto_recovery=True,
    )
    web_search = DuckDuckGoSearchAdapter(settings)
    web_reader = TrafilaturaReader(settings)
    source_policy = ResearchSourcePolicy.model_validate(
        settings.web_research_sources or {}
    )

    async def graph_factory(claim: ExecutionClaim) -> DurableConversationGraph:
        attempts = AttemptBoundary(coordinator.ledger, claim)
        request, head = await coordinator.load(claim)
        if (
            not isinstance(request, TurnRequestV2)
            or head.accepted_checkpoint_ref is None
        ):
            raise ValueError("V2 requires a versioned request and accepted root")
        record = await saver.aget_tuple(
            DurableConversationGraph.config(head.accepted_checkpoint_ref)
        )
        if record is None:
            raise ValueError("Accepted checkpoint is missing")
        DurableConversationGraph.require_supported_base(
            dict(record.metadata), head, {FLEXIBLE_GRAPH_VERSION}
        )
        accepted = await PostgresAcceptedCheckpointReader(saver).read(
            head.accepted_checkpoint_ref
        )
        if accepted.requirement_state is None:
            raise ValueError("Accepted canonical requirements are missing")
        binding = FlexiblePublicationInput.for_request(
            accepted, request, owner=await coordinator.owner(claim)
        )
        recovery = await coordinator.recovery_source(claim)
        original_context = None
        if binding.discard_pending:
            catalog = []
        elif recovery:
            source, _, _ = recovery
            source_record = await saver.aget_tuple(
                DurableConversationGraph.config(source)
            )
            if source_record is None:
                raise ValueError("Recovery checkpoint is missing")
            original_context = FlexibleExecutionContext.model_validate(
                source_record.checkpoint["channel_values"].get("publication_context")
            )
            if (
                original_context.context.run_id != claim.run_id
                or original_context.context.account_id != await coordinator.owner(claim)
                or original_context.context.requirements != accepted.requirement_state
            ):
                raise ValueError("Recovery publication context mismatch")
            catalog = original_context.context.catalog
        elif (
            accepted.flexible
            and accepted.flexible.pending_resolution
            and draft_needs_catalog(
                RequirementDraft(
                    items=accepted.flexible.pending_resolution.draft.resolution.draft.items
                )
            )
        ):
            catalog = await BackendRequirementCatalog(
                client, settings.backend_api_url, attempts, mapping
            ).load(RequirementDraft(items=[]), accepted.requirement_state)
        else:
            catalog = []
        products = BackendProductCatalog(client, settings.backend_api_url, attempts)
        pending = accepted.flexible
        fixed_tasks = (
            pending is not None
            and pending.status == "WAITING_INPUT"
            and pending.pending_task_resolution is None
            and request.pending_action is None
        )
        if request.selected_product_ids:
            if pending and pending.status == "WAITING_INPUT":
                raise ValueError(
                    "Finish or replace the pending task before selecting new products"
                )
            fixed_tasks = True
        tasks = (
            [item.task for item in pending.task_outcomes]
            if fixed_tasks and pending
            else []
        )
        if binding.discard_pending:
            tasks = []
            fixed_tasks = True
        if request.selected_product_ids:
            tasks = [
                CompletionRequirement(
                    kind="COMPARE"
                    if len(request.selected_product_ids) > 1
                    else "ANSWER",
                    product_ids=request.selected_product_ids,
                )
            ]
        if original_context:
            tasks = original_context.required
            fixed_tasks = not original_context.registered
        tool_context = (
            original_context.context
            if original_context
            else ToolContext(
                account_id=await coordinator.owner(claim),
                conversation_id=claim.conversation_id,
                run_id=claim.run_id,
                requirements=accepted.requirement_state,
                memories=await PostgresUserMemoryStore(engine).list(
                    await coordinator.owner(claim)
                ),
                read_catalog_allowed=True,
                execution_allowed=True,
                catalog=catalog,
                catalog_ready=bool(catalog),
                saved_build=SavedBuildEvidence(
                    source_ref=head.accepted_checkpoint_ref,
                    snapshot=accepted.public_build,
                )
                if accepted.public_build
                else None,
                history=await PostgresConversationStore(engine).dialogue(
                    await coordinator.owner(claim),
                    claim.conversation_id,
                    head.revision,
                ),
            )
        )
        requirement_extractor = NativeRequirementExtractor(provider, attempts)
        task_interpreter = NativeTaskExtractor(
            provider, attempts, context=tool_context
        )
        return DurableConversationGraph(
            build_flexible_graph(
                NativeFlexibleModel(provider),
                ToolDispatcher(
                    products,
                    service,
                    BudgetedWebSearch(web_search, attempts),
                    BudgetedWebReader(web_reader, attempts),
                    source_policy=source_policy,
                ),
                tool_context,
                tasks,
                attempts,
                conversation_revision=head.revision,
                publication=binding,
                task_catalog=None if fixed_tasks else products,
                checkpointer=saver,
                clarifier=requirement_extractor,
                unit_clarifier=requirement_extractor,
                task_clarifier=task_interpreter,
                amender=requirement_extractor,
                task_amender=task_interpreter,
                user_extractor=requirement_extractor,
                catalog_loader=BackendRequirementCatalog(
                    client, settings.backend_api_url, attempts, mapping
                ),
            ),
            accepted_versions={FLEXIBLE_GRAPH_VERSION},
        )

    executor = ManagedExecutor(
        coordinator,
        ConversationExecution(
            coordinator,
            graph_factory,
            FLEXIBLE_GRAPH_VERSION,
        ),
        worker_id=uuid4(),
    )
    try:
        await executor.start()
        yield StatefulRunService(coordinator, executor.wake, executor.cancel_run)
    finally:
        # Shutdown may exceed grace, but live CPU/checkpoint work must not lose
        # its clients/DB beneath it or be reported stopped merely on a timeout.
        while not await executor.stop(grace_seconds=30):
            logger.warning("Stateful shutdown is waiting for execution drain")
        await provider.aclose()
