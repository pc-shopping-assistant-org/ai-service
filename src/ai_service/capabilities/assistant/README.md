# Assistant capability

This slice owns deterministic request planning. It intentionally does not
retrieve catalog data or call an LLM: those responsibilities are application
ports injected by the composition root.

The graph output is validated before retrieval, and every product fact in a
response must still come from the canonical backend catalog.
