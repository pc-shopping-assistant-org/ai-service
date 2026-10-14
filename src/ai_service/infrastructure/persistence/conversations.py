"""Async owner-scoped reads/creation. No deletion before stop/drain + saver purge."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import insert, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_service.application.ports.stateful_conversation import (
    ConversationInitializer,
    ConversationNotFound,
    ConversationRecord,
    InvalidConversationCursor,
)
from ai_service.capabilities.assistant.limits import (
    MAX_DIALOGUE_CHARS,
    MAX_DIALOGUE_TURNS,
)
from ai_service.capabilities.assistant.public_contracts import MessageViewV1, PageV1
from ai_service.capabilities.assistant.react.contracts import (
    DialogueTurn,
)
from ai_service.capabilities.assistant.stateful_contracts import RunStatus
from ai_service.infrastructure.persistence.schema import (
    agent_runs,
    conversation_messages,
    conversations,
)


class PostgresConversationStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def dialogue(
        self,
        account_id: UUID,
        conversation_id: UUID,
        revision: int,
    ) -> list[DialogueTurn]:
        owned = await self.get(account_id, conversation_id)
        if owned.revision != revision:
            raise ValueError("Dialogue must match the accepted conversation revision")
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(
                            agent_runs.c.request_payload,
                            conversation_messages.c.content,
                        )
                        .join(
                            conversation_messages,
                            conversation_messages.c.run_id == agent_runs.c.id,
                        )
                        .where(
                            agent_runs.c.conversation_id == conversation_id,
                            agent_runs.c.kind == "CHAT",
                            agent_runs.c.status.in_(
                                [RunStatus.COMPLETED, RunStatus.WAITING_INPUT]
                            ),
                            agent_runs.c.base_revision < revision,
                            conversation_messages.c.role == "assistant",
                        )
                        .order_by(agent_runs.c.base_revision.desc())
                        .limit(MAX_DIALOGUE_TURNS)
                    )
                )
                .mappings()
                .all()
            )
        turns: list[DialogueTurn] = []
        size = 0
        for row in rows:
            turn = DialogueTurn(
                user_text=row["request_payload"]["message"],
                assistant_text=row["content"],
            )
            size += len(turn.user_text) + len(turn.assistant_text)
            if size > MAX_DIALOGUE_CHARS:
                break  # Keep complete recent turns; never cut a message mid-meaning.
            turns.append(turn)
        return list(reversed(turns))

    async def messages(
        self, account_id: UUID, conversation_id: UUID, limit: int, after: UUID | None
    ) -> PageV1[MessageViewV1]:
        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100")
        async with self.engine.connect() as connection:
            owned = await connection.scalar(
                select(conversations.c.id).where(
                    conversations.c.id == conversation_id,
                    conversations.c.account_id == account_id,
                )
            )
            if owned is None:
                raise ConversationNotFound()
            sequence = 0
            if after is not None:
                cursor_sequence = await connection.scalar(
                    select(conversation_messages.c.sequence).where(
                        conversation_messages.c.conversation_id == conversation_id,
                        conversation_messages.c.id == after,
                    )
                )
                if cursor_sequence is None:
                    raise InvalidConversationCursor()
                sequence = cursor_sequence
            rows = (
                (
                    await connection.execute(
                        select(conversation_messages)
                        .where(
                            conversation_messages.c.conversation_id == conversation_id,
                            conversation_messages.c.sequence > sequence,
                        )
                        .order_by(conversation_messages.c.sequence)
                        .limit(limit + 1)
                    )
                )
                .mappings()
                .all()
            )
        items = [
            MessageViewV1(
                message_id=row["id"],
                run_id=row["run_id"],
                sequence=row["sequence"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows[:limit]
        ]
        return PageV1[MessageViewV1](
            items=items,
            next_cursor=str(items[-1].message_id) if len(rows) > limit else None,
        )

    async def create(self, account_id: UUID, title: str | None) -> ConversationRecord:
        if title is not None:
            title = title.strip()
            if not title or len(title) > 200:
                raise ValueError("Title must have 1 to 200 characters")
        async with self.engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        insert(conversations)
                        .values(
                            id=uuid4(),
                            account_id=account_id,
                            title=title,
                        )
                        .returning(*conversations.c)
                    )
                )
                .mappings()
                .one()
            )
            return self._record(row)

    async def get(self, account_id: UUID, conversation_id: UUID) -> ConversationRecord:
        async with self.engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        select(conversations).where(
                            conversations.c.id == conversation_id,
                            conversations.c.account_id == account_id,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise ConversationNotFound()
            return self._record(row)

    async def initialize(
        self,
        account_id: UUID,
        conversation_id: UUID,
        initializer: ConversationInitializer,
    ) -> ConversationRecord:
        # One short row lock for native root creation; no model/catalog I/O here.
        async with self.engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        select(conversations)
                        .where(
                            conversations.c.id == conversation_id,
                            conversations.c.account_id == account_id,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise ConversationNotFound()
            if row["accepted_checkpoint_id"] is not None:
                return self._record(row)
            if row["revision"] != 0 or row["active_run_id"] is not None:
                raise ValueError("Conversation cannot be initialized while in use")
            for table in (agent_runs, conversation_messages):
                if (
                    await connection.execute(
                        select(table.c.id)
                        .where(
                            table.c.conversation_id == conversation_id,
                        )
                        .limit(1)
                    )
                ).first() is not None:
                    raise ValueError(
                        "Populated legacy conversation requires explicit migration"
                    )
            ref = await initializer.initialize(conversation_id)
            if ref.thread_id != conversation_id or ref.checkpoint_ns != "":
                raise ValueError("Initial checkpoint belongs to another conversation")
            accepted = (
                (
                    await connection.execute(
                        update(conversations)
                        .where(
                            conversations.c.id == conversation_id,
                            conversations.c.revision == 0,
                            conversations.c.accepted_checkpoint_id.is_(None),
                            conversations.c.active_run_id.is_(None),
                        )
                        .values(accepted_checkpoint_id=ref.checkpoint_id)
                        .returning(*conversations.c)
                    )
                )
                .mappings()
                .one()
            )
            return self._record(accepted)

    async def list_owned(
        self,
        account_id: UUID,
        limit: int = 20,
        before: tuple[datetime, UUID] | None = None,
    ) -> list[ConversationRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100")
        query = select(conversations).where(conversations.c.account_id == account_id)
        if before is not None:
            query = query.where(
                tuple_(conversations.c.updated_at, conversations.c.id) < before
            )
        query = query.order_by(
            conversations.c.updated_at.desc(), conversations.c.id.desc()
        ).limit(limit)
        async with self.engine.connect() as connection:
            return [
                self._record(row)
                for row in (await connection.execute(query)).mappings()
            ]

    @staticmethod
    def _record(row) -> ConversationRecord:
        return ConversationRecord.model_validate(
            {key: row[key] for key in ConversationRecord.model_fields}
        )
