"""V2 process lifecycle. No request-time schema creation or RAM fallback."""

from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from ai_service.application.ports.stateful_conversation import AccessTokenVerifier
from ai_service.application.ports.user_memory import UserMemoryStore
from ai_service.application.use_cases.run_events import StatefulRunEventService
from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
    StatefulRunService,
)
from ai_service.config.settings import Settings
from ai_service.infrastructure.auth import JwtAuthenticator
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.persistence.events import PostgresRunEventStore
from ai_service.infrastructure.persistence.runtime import open_stateful_database


@dataclass(frozen=True)
class StatefulResources:
    conversations: StatefulConversationService
    authenticator: AccessTokenVerifier
    runs: StatefulRunService | None = None
    events: StatefulRunEventService | None = None
    memories: UserMemoryStore | None = None


@asynccontextmanager
async def open_stateful_resources(settings: Settings):
    # Optional dependency is imported only when stateful mode actually starts.
    from sqlalchemy import select, text

    from ai_service.infrastructure.persistence.conversations import (
        PostgresConversationStore,
    )
    from ai_service.infrastructure.persistence.schema import metadata

    settings.require_stateful_configuration()
    async with open_stateful_database(settings) as (engine, saver):
        async with engine.connect() as connection:
            version = (
                await connection.execute(
                    text("SELECT version_num FROM alembic_version")
                )
            ).scalar_one()
            if version != "0006_user_memory":
                raise RuntimeError(
                    "AI application migration revision does not match this release"
                )
            for table in metadata.sorted_tables:
                await connection.execute(select(table).limit(0))
        # Readiness only; absent framework bootstrap must fail startup, not run setup().
        await saver.aget_tuple(
            {"configurable": {"thread_id": "__ai_readiness__", "checkpoint_ns": ""}}
        )
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            resources = StatefulResources(
                conversations=StatefulConversationService(
                    PostgresConversationStore(engine),
                    PostgresAcceptedCheckpointReader(saver),
                    NativeConversationInitializer(saver),
                ),
                authenticator=JwtAuthenticator(
                    client,
                    str(settings.jwt_jwks_url),
                    settings.jwt_issuer,
                    settings.jwt_audience,
                ),
            )
            from ai_service.infrastructure.execution.build_runtime import (
                open_build_runtime,
            )

            async with open_build_runtime(settings, engine, saver, client) as runs:
                from ai_service.infrastructure.persistence.user_memory import (
                    PostgresUserMemoryStore,
                )

                yield StatefulResources(
                    resources.conversations,
                    resources.authenticator,
                    runs,
                    StatefulRunEventService(PostgresRunEventStore(engine), runs.store),
                    PostgresUserMemoryStore(engine),
                )
