"""Owner-authorized public reads, independent of graph execution and DB drivers."""

from collections.abc import Callable
from uuid import UUID

from ai_service.application.ports.stateful_conversation import (
    AcceptedCheckpointReader,
    ConversationInitializer,
    ConversationNotFound,
    ConversationRecord,
    InvalidConversationCursor,
    StatefulConversationStore,
    StatefulRunStore,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.public_contracts import (
    ConversationViewV1,
    MessageViewV1,
    PageV1,
    RunViewV1,
    to_public_consultation,
)
from ai_service.capabilities.assistant.react.public_views import (
    ConversationViewV2,
    to_public_flexible,
)
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequest
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
)


class StatefulConversationService:
    def __init__(
        self,
        store: StatefulConversationStore,
        reader: AcceptedCheckpointReader,
        initializer: ConversationInitializer,
        *,
        view_version: int = 1,
    ) -> None:
        self.store = store
        self.reader = reader
        self.initializer = initializer
        if view_version not in (1, 2):
            raise ValueError("Unknown public conversation version")
        self.view_version = view_version

    async def create(self, account_id: UUID, title: str | None) -> ConversationViewV1:
        record = await self.store.create(account_id, title)
        record = await self.store.initialize(account_id, record.id, self.initializer)
        return await self._view(record)

    async def get(self, account_id: UUID, conversation_id: UUID) -> ConversationViewV1:
        # Authorize before any framework lookup. No thread selector from HTTP.
        return await self._view(await self.store.get(account_id, conversation_id))

    async def list(
        self, account_id: UUID, limit: int = 20, cursor: str | None = None
    ) -> PageV1[ConversationViewV1] | PageV1[ConversationViewV2]:
        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100")
        before = None
        if cursor is not None:
            try:
                cursor_id = UUID(cursor)
            except ValueError:
                raise InvalidConversationCursor() from None
            try:
                previous = await self.store.get(account_id, cursor_id)
            except ConversationNotFound:
                raise InvalidConversationCursor() from None
            before = (previous.updated_at, previous.id)
        rows = await self.store.list_owned(account_id, limit, before)
        more = False
        if len(rows) == limit:
            last = rows[-1]
            more = bool(
                await self.store.list_owned(account_id, 1, (last.updated_at, last.id))
            )
        views = [await self._view(row) for row in rows]
        next_cursor = str(rows[-1].id) if more else None
        if self.view_version == 2:
            return PageV1[ConversationViewV2](
                items=[
                    ConversationViewV2.model_validate(view.model_dump())
                    for view in views
                ],
                next_cursor=next_cursor,
            )
        return PageV1[ConversationViewV1](items=views, next_cursor=next_cursor)

    async def messages(
        self, account_id: UUID, conversation_id: UUID, limit: int, cursor: str | None
    ) -> PageV1[MessageViewV1]:
        try:
            after = UUID(cursor) if cursor is not None else None
        except ValueError:
            raise InvalidConversationCursor() from None
        return await self.store.messages(account_id, conversation_id, limit, after)

    async def _view(self, record: ConversationRecord) -> ConversationViewV1:
        state = ConsultationStateV1()
        build = None
        clarification = None
        task_clarification = None
        flexible = None
        if record.accepted_checkpoint_id is not None:
            snapshot = await self.reader.read(
                CheckpointRefV1(
                    thread_id=record.id,
                    checkpoint_id=record.accepted_checkpoint_id,
                )
            )
            state, build = snapshot.consultation, snapshot.public_build
            clarification = snapshot.public_clarification
            task_clarification = snapshot.public_task_clarification
            if self.view_version == 2 and snapshot.flexible is not None:
                if snapshot.requirement_state is None:
                    raise ValueError("V2 public state requires canonical requirements")
                flexible = to_public_flexible(
                    snapshot.flexible, snapshot.requirement_state
                )
        elif record.revision != 0:
            raise ValueError("Published revision has no accepted checkpoint")
        view = ConversationViewV1(
            conversation_id=record.id,
            readiness="READY"
            if record.accepted_checkpoint_id is not None
            else "INITIALIZING",
            revision=record.revision,
            title=record.title,
            updated_at=record.updated_at,
            state=to_public_consultation(state),
            build=build,
            clarification=clarification,
            task_clarification=task_clarification,
        )
        return (
            ConversationViewV2.model_validate(
                view.model_dump() | {"flexible": flexible}
            )
            if self.view_version == 2
            else view
        )


class StatefulRunService:
    """Commit admission before waking the process-owned executor; no HTTP task."""

    def __init__(
        self,
        store: StatefulRunStore,
        wake: Callable[[], None],
        cancel_signal: Callable[[UUID], bool] | None = None,
    ) -> None:
        self.store, self.wake = store, wake
        self.cancel_signal = cancel_signal

    async def cancel(
        self, owner: UUID, conversation_id: UUID, run_id: UUID
    ) -> RunViewV1:
        result = await self.store.cancel(owner, conversation_id, run_id)
        if self.cancel_signal is not None:
            self.cancel_signal(run_id)
        self.wake()
        return result

    async def submit(self, owner: UUID, request: TurnRequest) -> RunViewV1:
        result = await self.store.submit(owner, request)
        self.wake()
        return result

    async def get(self, owner: UUID, conversation_id: UUID, run_id: UUID) -> RunViewV1:
        return await self.store.get(owner, conversation_id, run_id)
