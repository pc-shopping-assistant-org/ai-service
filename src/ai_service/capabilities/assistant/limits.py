"""Shared assistant contract limits; transport defaults live with their adapter."""

MAX_TURN_MESSAGE_CHARS = 4_000
MIN_COMPARISON_PRODUCTS = 2
MAX_SELECTED_PRODUCTS = 5
MAX_PUBLIC_PRODUCTS = 20
MAX_DIALOGUE_TURNS = 4
MAX_DIALOGUE_CHARS = 12_000
# Eight-part research needs up to sixteen external requests, plus tool admission
# and catalog/model-adjacent reads. Persist this bounded allowance for each new run.
DEFAULT_TOOL_IO_LIMIT = 48
