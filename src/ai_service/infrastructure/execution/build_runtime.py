"""Managed guided/ReAct composition; one selected graph, one executor lifecycle."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_service.application.call_attempts import AttemptBoundary, ExecutionClaim
from ai_service.application.use_cases.stateful_conversations import StatefulRunService
from ai_service.capabilities.assistant.react.contracts import (
    SavedBuildEvidence,
    ToolContext,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexibleExecutionContext,
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.config.enums import StatefulReactVersion
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
from ai_service.infrastructure.graph.requirement_build import (
    ReactComposition,
    build_requirement_graph,
)
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.providers.factory import build_model_provider
from ai_service.infrastructure.providers.react_model import (
    NativeFlexibleModel,
    NativeReactModel,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)
from ai_service.infrastructure.search.duckduckgo_adapter import DuckDuckGoSearchAdapter
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader

from .conversation import ConversationExecution
from .supervisor import ManagedExecutor

GRAPH_VERSION = "native-guided-build-v5-amendment"
REACT_GRAPH_VERSION = "native-stateful-react-v6-task-amendment"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_build_runtime(
    settings: Settings,
    engine: AsyncEngine,
    saver: BaseCheckpointSaver,
    client: httpx.AsyncClient,
):
    mapping = HardwareCatalogMapping.model_validate(settings.hardware_catalog_mapping)
    provider = build_model_provider(settings)
    if provider is None:
        raise ValueError("Stateful execution requires a model provider")
    flexible = settings.stateful_react_version == StatefulReactVersion.V2
    service = PCBuildApplicationService()
    coordinator = PostgresRunCoordinator(
        engine,
        saver,
        request_versions=(2,) if flexible else (1,),
        flexible_service=service if flexible else None,
        max_attempts=2 if flexible else 1,
        auto_recovery=flexible,
    )
    web_search = (
        DuckDuckGoSearchAdapter(settings) if settings.stateful_react_enabled else None
    )
    web_reader = (
        TrafilaturaReader(settings) if settings.stateful_react_enabled else None
    )

    async def graph_factory(claim: ExecutionClaim) -> DurableConversationGraph:
        attempts = AttemptBoundary(coordinator.ledger, claim)
        extractor = NativeRequirementExtractor(provider, attempts)
        if flexible:
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
                    source_record.checkpoint["channel_values"].get(
                        "publication_context"
                    )
                )
                if (
                    original_context.context.run_id != claim.run_id
                    or original_context.context.account_id
                    != await coordinator.owner(claim)
                    or original_context.context.requirements
                    != accepted.requirement_state
                ):
                    raise ValueError("Recovery publication context mismatch")
                catalog = original_context.context.catalog
            else:
                catalog = await BackendRequirementCatalog(
                    client, settings.backend_api_url, attempts, mapping
                ).load(Draft(items=[]), accepted.requirement_state)
            products = BackendProductCatalog(client, settings.backend_api_url, attempts)
            pending = accepted.flexible
            fixed_tasks = (
                pending is not None
                and pending.status == "WAITING_INPUT"
                and pending.pending_task_resolution is None
                and request.pending_action is None
            )
            tasks = (
                [item.task for item in pending.task_outcomes]
                if fixed_tasks and pending
                else []
            )
            if binding.discard_pending:
                tasks = []
                fixed_tasks = True
            if original_context:
                tasks = original_context.required
                fixed_tasks = not original_context.registered
            return DurableConversationGraph(
                build_flexible_graph(
                    NativeFlexibleModel(provider),
                    ToolDispatcher(products, service, web_search, web_reader),
                    original_context.context
                    if original_context
                    else ToolContext(
                        account_id=await coordinator.owner(claim),
                        conversation_id=claim.conversation_id,
                        run_id=claim.run_id,
                        requirements=accepted.requirement_state,
                        read_catalog_allowed=True,
                        execution_allowed=True,
                        catalog=catalog,
                        catalog_ready=True,
                        saved_build=SavedBuildEvidence(
                            source_ref=head.accepted_checkpoint_ref,
                            snapshot=accepted.public_build,
                        )
                        if accepted.public_build
                        else None,
                    ),
                    tasks,
                    attempts,
                    conversation_revision=head.revision,
                    publication=binding,
                    task_extractor=None if fixed_tasks else extractor,
                    task_catalog=None if fixed_tasks else products,
                    checkpointer=saver,
                    clarifier=extractor,
                    unit_clarifier=extractor,
                    task_clarifier=extractor,
                    amender=extractor,
                    task_amender=extractor,
                    user_extractor=extractor,
                ),
                accepted_versions={FLEXIBLE_GRAPH_VERSION},
            )
        react = None
        if settings.stateful_react_enabled:
            _, head = await coordinator.load(claim)
            if head.accepted_checkpoint_ref is None:
                raise ValueError("ReAct requires an initialized accepted checkpoint")
            record = await saver.aget_tuple(
                DurableConversationGraph.config(head.accepted_checkpoint_ref)
            )
            if record is None:
                raise ValueError("Accepted checkpoint is missing")
            snapshot = record.checkpoint["channel_values"].get("public_build")
            saved = (
                SavedBuildEvidence(
                    source_ref=head.accepted_checkpoint_ref, snapshot=snapshot
                )
                if snapshot is not None
                else None
            )
            products = BackendProductCatalog(client, settings.backend_api_url, attempts)
            react = ReactComposition(
                NativeReactModel(provider),
                ToolDispatcher(
                    products, PCBuildApplicationService(), web_search, web_reader
                ),
                ToolContext(
                    account_id=await coordinator.owner(claim),
                    conversation_id=claim.conversation_id,
                    run_id=claim.run_id,
                    requirements=CanonicalRequirementState(),
                    read_catalog_allowed=True,
                    execution_allowed=True,
                    catalog_ready=False,
                    saved_build=saved,
                ),
                products,
                attempts,
                task_clarifier=extractor,
                task_amender=extractor,
            )
        return DurableConversationGraph(
            build_requirement_graph(
                extractor,
                BackendRequirementCatalog(
                    client, settings.backend_api_url, attempts, mapping
                ),
                checkpointer=saver,
                clarifier=extractor,
                unit_clarifier=extractor,
                amender=extractor,
                react=react,
            )
        )

    executor = ManagedExecutor(
        coordinator,
        ConversationExecution(
            coordinator,
            graph_factory,
            FLEXIBLE_GRAPH_VERSION
            if flexible
            else REACT_GRAPH_VERSION
            if settings.stateful_react_enabled
            else GRAPH_VERSION,
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
