"""V5 makes event payload identity checks total: missing JSON fields reject."""

from sqlalchemy import CheckConstraint, MetaData

from .schema_v4 import metadata as v4

metadata = MetaData(naming_convention=v4.naming_convention)
for existing in v4.tables.values():
    existing.to_metadata(metadata)
events = metadata.tables["agent_run_events"]
old = next(
    item for item in events.constraints if item.name == "ck_agent_run_events_payload"
)
events.constraints.remove(old)
EVENT_PAYLOAD_CHECK = """COALESCE(
    jsonb_typeof(payload) = 'object'
    AND jsonb_typeof(payload->'data') = 'object'
    AND jsonb_typeof(payload->'errors') = 'array'
    AND jsonb_typeof(payload->'message') = 'string'
    AND payload->'data'->>'event' = event_type
    AND payload->'data'->>'sequence' = sequence::text
    AND payload->'data'->>'run_id' = run_id::text
    AND payload->'data'->>'conversation_id' = conversation_id::text,
    false
)"""
events.append_constraint(CheckConstraint(EVENT_PAYLOAD_CHECK, name="payload"))
