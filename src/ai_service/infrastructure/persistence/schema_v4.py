"""Frozen V4 adds application-owned, ordered run events, not checkpoint writes."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKeyConstraint,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

from .schema_v3 import metadata as v3

metadata = MetaData(naming_convention=v3.naming_convention)
for existing in v3.tables.values():
    existing.to_metadata(metadata)

agent_run_events = Table(
    "agent_run_events",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("conversation_id", Uuid, nullable=False),
    Column("run_id", Uuid, nullable=False),
    Column("sequence", BigInteger, nullable=False),
    Column("dedup_key", String(200), nullable=False),
    Column("event_type", String(20), nullable=False),
    Column("payload", JSONB, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    ForeignKeyConstraint(
        ["conversation_id", "run_id"],
        ["agent_runs.conversation_id", "agent_runs.id"],
        ondelete="CASCADE",
    ),
    UniqueConstraint("run_id", "sequence", name="uq_agent_run_events_sequence"),
    UniqueConstraint("run_id", "dedup_key", name="uq_agent_run_events_dedup"),
    CheckConstraint("sequence > 0", name="sequence"),
    CheckConstraint("length(btrim(dedup_key)) > 0", name="dedup_key"),
    CheckConstraint(
        "event_type IN ('START','DELTA','COMPLETED','ERROR')", name="event_type"
    ),
    CheckConstraint(
        "jsonb_typeof(payload) = 'object' AND payload->'data'->>'event' = event_type",
        name="payload",
    ),
)
