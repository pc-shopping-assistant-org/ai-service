"""Reject absent/mismatched event envelope identities without deleting history."""

from alembic import op

from ai_service.infrastructure.persistence.schema_v5 import EVENT_PAYLOAD_CHECK

revision = "0005_event_payload"
down_revision = "0004_run_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_agent_run_events_payload"), "agent_run_events", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_agent_run_events_payload"), "agent_run_events", EVENT_PAYLOAD_CHECK
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_agent_run_events_payload"), "agent_run_events", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_agent_run_events_payload"),
        "agent_run_events",
        "jsonb_typeof(payload) = 'object' AND payload->'data'->>'event' = event_type",
    )
