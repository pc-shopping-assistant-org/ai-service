"""V3 permits an accepted native initial root at revision zero."""

from sqlalchemy import CheckConstraint, MetaData

from .schema_v2 import metadata as v2

metadata = MetaData(naming_convention=v2.naming_convention)
for table in v2.tables.values():
    table.to_metadata(metadata)
conversations = metadata.tables["conversations"]
old = next(
    c for c in conversations.constraints if c.name == "ck_conversations_accepted_head"
)
conversations.constraints.remove(old)
conversations.append_constraint(
    CheckConstraint(
        "revision = 0 OR accepted_checkpoint_id IS NOT NULL",
        name="accepted_root_head",
    )
)
