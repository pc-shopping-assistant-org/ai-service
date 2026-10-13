"""Frozen V2: durable external-attempt budgets, not tool-result evidence."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    Uuid,
    text,
)

from .schema_v1 import metadata as v1

metadata = MetaData(naming_convention=v1.naming_convention)
for existing in v1.tables.values():
    existing.to_metadata(metadata)

agent_call_budgets = Table(
    "agent_call_budgets",
    metadata,
    Column(
        "run_id",
        Uuid,
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("model_limit", Integer, nullable=False),
    Column("tool_io_limit", Integer, nullable=False),
    Column("deadline", DateTime(timezone=True), nullable=False),
    CheckConstraint("model_limit >= 0 AND tool_io_limit >= 0", name="limits"),
)

agent_call_attempts = Table(
    "agent_call_attempts",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column(
        "run_id",
        Uuid,
        ForeignKey("agent_call_budgets.run_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("execution_id", Uuid, nullable=False),
    Column("worker_instance_id", Uuid, nullable=False),
    Column("generation", BigInteger, nullable=False),
    Column("kind", String(10), nullable=False),
    Column("operation_key", String(200), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("attempt_number", Integer, nullable=False),
    Column("status", String(10), nullable=False, server_default="RESERVED"),
    Column(
        "reserved_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint(
        "run_id", "operation_key", "attempt_number", name="uq_call_attempt_operation"
    ),
    CheckConstraint("generation > 0 AND attempt_number > 0", name="positive_identity"),
    CheckConstraint("kind IN ('MODEL','TOOL_IO')", name="kind"),
    CheckConstraint(
        "status IN ('RESERVED','SUCCEEDED','FAILED','UNKNOWN')", name="status"
    ),
    CheckConstraint("btrim(operation_key) <> ''", name="operation_key"),
    CheckConstraint("request_hash ~ '^[0-9a-f]{64}$'", name="request_hash"),
    CheckConstraint("(status = 'RESERVED') = (finished_at IS NULL)", name="finished"),
)
Index(
    "ix_call_attempt_run_kind", agent_call_attempts.c.run_id, agent_call_attempts.c.kind
)
