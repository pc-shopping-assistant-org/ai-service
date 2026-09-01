# Capability slices

New AI features should be added as a vertical slice under this package rather
than as another global `service.py` or `utils.py` module:

```text
capabilities/<feature>/
├── domain.py       # pure business state/value rules (optional)
├── schemas.py      # feature input/output models (transport-neutral)
├── graph.py        # Pydantic Graph definition, when orchestration is needed
├── use_case.py     # application orchestration; depends on ports only
├── tools.py        # PydanticAI tools, when the feature needs tools
└── README.md       # capability invariants and flow
```

The HTTP layer may map the feature schemas to the public API envelope, but it
must not construct an agent, provider, graph or database client. Put those
bindings in `infrastructure/composition.py` and expose only an inbound port to
the route.

The current assistant graph definitions live in
`capabilities/assistant/graphs/`. The old `ai_service.graphs` modules remain
thin compatibility re-exports while existing integrations migrate.
