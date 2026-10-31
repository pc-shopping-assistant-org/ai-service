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

# Complete canonical scans, including non-optimizer products, remain bounded.
MAX_HARDWARE_CATALOG_ROWS = 3_000
HARDWARE_CATALOG_PAGE_SIZE = 100
MAX_HARDWARE_CATALOG_PAGES = 30
# Canonical catalog/provenance stays in checkpoints, never the model prompt.
MAX_REACT_STATE_BYTES = 16_000_000
MAX_BUILD_RESULT_BYTES = 8_000_000
