"""Allow revision-zero accepted root; preserve existing rows and history."""

from alembic import op

revision = "0003_initial_root"
down_revision = "0002_call_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_conversations_accepted_head"), "conversations", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_conversations_accepted_root_head"),
        "conversations",
        "revision = 0 OR accepted_checkpoint_id IS NOT NULL",
    )


def downgrade() -> None:
    # Refuse downgrade if roots exist; never silently remove accepted state.
    op.create_check_constraint(
        op.f("ck_conversations_accepted_head"),
        "conversations",
        "(revision = 0) = (accepted_checkpoint_id IS NULL)",
    )
    op.drop_constraint(
        op.f("ck_conversations_accepted_root_head"), "conversations", type_="check"
    )
