"""Account-scoped explicit advisory memory, separate from conversation state."""

from alembic import op

from ai_service.infrastructure.persistence.schema_v6 import metadata

revision = "0006_user_memory"
down_revision = "0005_event_payload"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata.tables["user_memories"].create(op.get_bind())


def downgrade() -> None:
    metadata.tables["user_memories"].drop(op.get_bind())
