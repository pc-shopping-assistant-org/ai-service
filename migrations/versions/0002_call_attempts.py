"""Add conservative durable external-attempt accounting without editing V1."""

from alembic import op

from ai_service.infrastructure.persistence.schema_v2 import (
    agent_call_attempts,
    agent_call_budgets,
    metadata,
)

revision = "0002_call_attempts"
down_revision = "0001_stateful"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata.create_all(
        op.get_bind(),
        tables=[agent_call_budgets, agent_call_attempts],
        checkfirst=False,
    )


def downgrade() -> None:
    metadata.drop_all(
        op.get_bind(),
        tables=[agent_call_attempts, agent_call_budgets],
        checkfirst=False,
    )
