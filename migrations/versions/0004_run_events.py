"""Add durable run event journal; preserve application and native history."""

from alembic import op

from ai_service.infrastructure.persistence.schema_v4 import agent_run_events

revision = "0004_run_events"
down_revision = "0003_initial_root"
branch_labels = None
depends_on = None


def upgrade() -> None:
    agent_run_events.create(op.get_bind())


def downgrade() -> None:
    # Do not discard published reconnect history silently.
    raise RuntimeError(
        "Run-event downgrade requires an explicit history retention decision"
    )
