"""Current application metadata; framework checkpoint tables are never included."""

from ai_service.infrastructure.persistence.schema_v6 import metadata

# Current copies, not frozen historical migration metadata.
agent_call_attempts = metadata.tables["agent_call_attempts"]
agent_call_budgets = metadata.tables["agent_call_budgets"]
agent_runs = metadata.tables["agent_runs"]
agent_run_events = metadata.tables["agent_run_events"]
conversations = metadata.tables["conversations"]
conversation_messages = metadata.tables["conversation_messages"]
user_memories = metadata.tables["user_memories"]

__all__ = [
    "agent_call_attempts",
    "agent_call_budgets",
    "agent_run_events",
    "agent_runs",
    "conversation_messages",
    "conversations",
    "metadata",
    "user_memories",
]
