"""Core schema constraints; real DB enforcement lives in integration_tests."""

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from ai_service.infrastructure.persistence.schema import metadata


def test_application_schema_preserves_scoped_execution_and_message_invariants():
    sql = "\n".join(
        str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in metadata.sorted_tables
    )
    assert set(metadata.tables) == {
        "conversations",
        "agent_runs",
        "conversation_messages",
        "agent_call_budgets",
        "agent_call_attempts",
        "agent_run_events",
    }
    assert "revision = 0 OR accepted_checkpoint_id IS NOT NULL" in sql
    assert "UNIQUE (conversation_id, request_id)" in sql
    assert "FOREIGN KEY(conversation_id, run_id)" in sql
    assert "UNIQUE (run_id, role)" in sql
    assert "FAILED_RETRYABLE" in sql and "FAILED_TERMINAL" in sql
    slot = next(c for c in metadata.tables["conversations"].foreign_key_constraints)
    assert slot.deferrable and slot.initially == "DEFERRED" and slot.use_alter
    event_sql = str(
        CreateTable(metadata.tables["agent_run_events"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "UNIQUE (run_id, sequence)" in event_sql
    assert "UNIQUE (run_id, dedup_key)" in event_sql
    assert "COALESCE(" in event_sql and "false" in event_sql
    assert "payload->'data'->>'sequence' = sequence::text" in event_sql
    assert "REFERENCES agent_runs (conversation_id, id) ON DELETE CASCADE" in event_sql


def test_migration_uses_application_metadata_not_framework_tables():
    assert not any("checkpoint" in table for table in metadata.tables)


def test_call_budget_ddl_preserves_attempts_and_cascades_without_changing_v1():
    from ai_service.infrastructure.persistence.schema_v1 import metadata as original

    assert len(original.tables) == 3
    sql = str(
        CreateTable(metadata.tables["agent_call_attempts"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "UNIQUE (run_id, operation_key, attempt_number)" in sql
    assert "UNKNOWN" in sql and "RESERVED" in sql
    assert "ON DELETE CASCADE" in sql
    assert "attempt_number > 0" in sql
    budget_sql = str(
        CreateTable(metadata.tables["agent_call_budgets"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "model_limit >= 0" in budget_sql and "tool_io_limit >= 0" in budget_sql
