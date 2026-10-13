"""Immutable V1 AI application tables (native saver schema is separate)."""

from alembic import op
from sqlalchemy import MetaData

from ai_service.infrastructure.persistence.schema_v1 import (
    metadata as application_metadata,
)

revision = "0001_stateful"
down_revision = None
branch_labels = None
depends_on = None


# This revision imports immutable V1, never current-schema metadata.
def schema() -> MetaData:
    snapshot = MetaData()
    for table in application_metadata.tables.values():
        table.to_metadata(snapshot)
    return snapshot


def upgrade() -> None:
    schema().create_all(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    schema().drop_all(op.get_bind(), checkfirst=False)
