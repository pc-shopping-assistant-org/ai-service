"""Explicit cross-conversation account memory; historical schemas stay frozen."""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Uuid,
    text,
)

from .schema_v5 import metadata as v5

metadata = MetaData(naming_convention=v5.naming_convention)
for existing in v5.tables.values():
    existing.to_metadata(metadata)

Table(
    "user_memories",
    metadata,
    Column("account_id", Uuid, primary_key=True),
    Column("key", String(64), primary_key=True),
    Column("kind", String(20), nullable=False),
    Column("content", String(400), nullable=False),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("key ~ '^[a-z][a-z0-9_-]{0,63}$'", name="key"),
    CheckConstraint("kind IN ('PREFERENCE', 'OWNED_CONTEXT')", name="kind"),
    CheckConstraint("length(btrim(content)) BETWEEN 1 AND 400", name="content"),
    CheckConstraint("expires_at > updated_at", name="expiry"),
)
