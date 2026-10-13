"""Current application metadata; framework checkpoint tables are never included."""

from ai_service.infrastructure.persistence.schema_v3 import metadata

# Current copies, not frozen historical migration metadata.
agent_call_attempts = metadata.tables["agent_call_attempts"]
agent_call_budgets = metadata.tables["agent_call_budgets"]
agent_runs = metadata.tables["agent_runs"]
conversations = metadata.tables["conversations"]
conversation_messages = metadata.tables["conversation_messages"]

__all__ = [
    "agent_call_attempts",
    "agent_call_budgets",
    "agent_runs",
    "conversation_messages",
    "conversations",
    "metadata",
]
