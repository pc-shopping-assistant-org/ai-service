"""Owner-scoped PostgreSQL memory; quota and upsert share one transaction."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_service.application.ports.user_memory import (
    MAX_MEMORIES,
    MEMORY_RETENTION_DAYS,
    MemoryEntry,
    MemoryInput,
    MemoryKey,
    MemoryLimitReached,
)

from .schema import user_memories as table


class PostgresUserMemoryStore:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def list(
        self, owner: UUID, *, at: datetime | None = None
    ) -> list[MemoryEntry]:
        async with self.engine.connect() as conn:
            records = (
                (
                    await conn.execute(
                        select(table)
                        .where(
                            table.c.account_id == owner,
                            table.c.expires_at > (at or datetime.now(UTC)),
                        )
                        .order_by(table.c.key)
                        .limit(MAX_MEMORIES)
                    )
                )
                .mappings()
                .all()
            )
        return [
            MemoryEntry.model_validate(dict(row), extra="ignore") for row in records
        ]

    async def put(self, owner: UUID, key: str, value: MemoryInput) -> MemoryEntry:
        key = TypeAdapter(MemoryKey).validate_python(key)
        value = MemoryInput.model_validate(value.model_dump())
        now = datetime.now(UTC)
        row = MemoryEntry(
            **value.model_dump(),
            key=key,
            updated_at=now,
            expires_at=now + timedelta(days=MEMORY_RETENTION_DAYS),
        )
        async with self.engine.begin() as conn:
            # Per-account lock serializes quota checking, not unrelated users.
            lock = owner.int & ((1 << 63) - 1)
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(:owner)"), {"owner": lock}
            )
            await conn.execute(
                delete(table).where(
                    table.c.account_id == owner, table.c.expires_at <= now
                )
            )
            existing = await conn.scalar(
                select(table.c.key).where(
                    table.c.account_id == owner, table.c.key == key
                )
            )
            count = await conn.scalar(
                select(func.count())
                .select_from(table)
                .where(table.c.account_id == owner)
            )
            if existing is None and (count or 0) >= MAX_MEMORIES:
                raise MemoryLimitReached()
            await conn.execute(
                insert(table)
                .values(account_id=owner, **row.model_dump())
                .on_conflict_do_update(
                    index_elements=[table.c.account_id, table.c.key],
                    set_=row.model_dump(exclude={"key"}),
                )
            )
        return row

    async def delete(self, owner: UUID, key: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                delete(table).where(table.c.account_id == owner, table.c.key == key)
            )
