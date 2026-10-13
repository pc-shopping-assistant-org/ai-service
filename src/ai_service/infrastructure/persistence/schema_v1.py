"""Frozen V1 DDL. Do not edit after release; evolve schema in a new migration."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData(
    naming_convention={
        "pk": "pk_%(table_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "ix": "ix_%(table_name)s_%(column_0_name)s",
    }
)


def timestamps() -> list[Column]:
    return [
        Column(
            name, DateTime(timezone=True), nullable=False, server_default=text("now()")
        )
        for name in ("created_at", "updated_at")
    ]


conversations = Table(
    "conversations",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("account_id", Uuid, nullable=False),
    Column("title", String(200)),
    Column("revision", BigInteger, nullable=False, server_default="0"),
    Column("accepted_checkpoint_id", Text),
    Column("active_run_id", Uuid),
    *timestamps(),
    CheckConstraint("revision >= 0", name="revision"),
    CheckConstraint(
        "(revision = 0) = (accepted_checkpoint_id IS NULL)", name="accepted_head"
    ),
    CheckConstraint(
        "accepted_checkpoint_id IS NULL OR btrim(accepted_checkpoint_id) <> ''",
        name="checkpoint_nonblank",
    ),
    ForeignKeyConstraint(
        ["id", "active_run_id"],
        ["agent_runs.conversation_id", "agent_runs.id"],
        name="fk_conversations_active_run",
        use_alter=True,
        deferrable=True,
        initially="DEFERRED",
    ),
)
Index(
    "ix_conversations_owner_list",
    conversations.c.account_id,
    conversations.c.updated_at.desc(),
    conversations.c.id.desc(),
)

agent_runs = Table(
    "agent_runs",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column(
        "conversation_id",
        Uuid,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("request_id", Uuid, nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("request_payload", JSONB, nullable=False),
    Column("kind", String(10), nullable=False),
    Column("status", String(30), nullable=False),
    Column("base_revision", BigInteger, nullable=False),
    *[
        Column(name, JSONB)
        for name in (
            "base_checkpoint_ref",
            "last_checkpoint_ref",
            "final_checkpoint_ref",
            "executor_stop_evidence",
            "versions",
            "error",
            "replay_snapshot",
        )
    ],
    Column(
        "authorized_recovery_source_refs",
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    ),
    Column("execution_id", Uuid),
    Column("worker_instance_id", Uuid),
    Column("lease_generation", BigInteger, nullable=False, server_default="0"),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("heartbeat_at", DateTime(timezone=True)),
    Column("cancel_requested", Boolean, nullable=False, server_default=text("false")),
    Column("attempt_count", Integer, nullable=False),
    Column("max_attempts", Integer, nullable=False),
    Column("retry_policy_version", Text, nullable=False),
    Column("retry_deadline", DateTime(timezone=True), nullable=False),
    Column("trace_id", Text),
    *timestamps(),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("conversation_id", "id", name="uq_agent_runs_scoped_id"),
    UniqueConstraint("conversation_id", "request_id", name="uq_agent_runs_request"),
    CheckConstraint("request_hash ~ '^[0-9a-f]{64}$'", name="request_hash"),
    CheckConstraint("jsonb_typeof(request_payload) = 'object'", name="request_payload"),
    CheckConstraint(
        "jsonb_typeof(authorized_recovery_source_refs) = 'array'", name="recovery_refs"
    ),
    CheckConstraint("kind IN ('CHAT','DEBUG','REPLAY')", name="kind"),
    CheckConstraint(
        "status IN ('PENDING','RUNNING','RECOVERING','WAITING_INPUT','COMPLETED','FAILED_RETRYABLE','FAILED_TERMINAL','CANCELLED')",
        name="status",
    ),
    CheckConstraint("base_revision >= 0 AND lease_generation >= 0", name="generations"),
    CheckConstraint("attempt_count >= 0 AND max_attempts > 0", name="attempts"),
)
Index(
    "ix_agent_runs_conversation",
    agent_runs.c.conversation_id,
    agent_runs.c.created_at,
    agent_runs.c.id,
)
Index("ix_agent_runs_lease", agent_runs.c.status, agent_runs.c.lease_expires_at)

conversation_messages = Table(
    "conversation_messages",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column(
        "conversation_id",
        Uuid,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("run_id", Uuid, nullable=False),
    Column("sequence", BigInteger, nullable=False),
    Column("role", String(10), nullable=False),
    Column("content", Text, nullable=False),
    Column("payload", JSONB),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    ForeignKeyConstraint(
        ["conversation_id", "run_id"],
        ["agent_runs.conversation_id", "agent_runs.id"],
        name="fk_messages_scoped_run",
        ondelete="CASCADE",
    ),
    UniqueConstraint("conversation_id", "sequence", name="uq_messages_sequence"),
    UniqueConstraint("run_id", "role", name="uq_messages_run_role"),
    CheckConstraint("sequence > 0", name="sequence"),
    CheckConstraint("role IN ('user','assistant')", name="role"),
    CheckConstraint("btrim(content) <> ''", name="content"),
)
