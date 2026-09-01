# AI service foundation

The service is organized around capability vertical slices and explicit clean
architecture boundaries. The current assistant is the first slice; future
catalog search, guided selection, evaluation, comparison and workflow
capabilities should follow the same shape.

```text
api/                         interface adapters (HTTP, SSE, validation)
  ↓ depends on
application/use_cases/       application orchestration
application/ports/           inbound and outbound contracts
  ↑ implemented by
infrastructure/              HTTP, model, graph and retrieval adapters
capabilities/<feature>/      feature-owned schemas, graphs and tools
```

`infrastructure/composition.py` is the composition root. It creates the
process-scoped adapters, PydanticAI answer generator, graph runners and
application use case. FastAPI receives the inbound `AssistantUseCase` through
`api/dependencies.py`; routes do not construct providers or graphs.
Runtime provider/retrieval choices and assistant branches use typed enums whose
values remain compatible with the environment/API strings.

## Rules for a new capability

1. Define feature input/output/state models in the capability package.
2. Put deterministic orchestration in a Pydantic Graph only when the flow has
   multiple validated steps or branches.
3. Keep the use case dependent on `application.ports`, never on an SDK, HTTP
   client, Qdrant implementation or provider secret.
4. Add a small outbound port for every external dependency and bind it in the
   composition root.
5. Keep PydanticAI agents/tools in the capability or infrastructure adapter;
   validate model output before mapping it to the API envelope.
6. Expose stable `ApiResponse`/SSE envelopes only from the interface adapter.
7. Test the capability with fake ports first, then run the full suite and
   document any unresolved requirement in `USECASE_IMPLEMENTATION.md`.

## Current compatibility seams

- `services/assistant_service.py` is a legacy constructor that supplies local
  defaults for existing scripts/tests. Production wiring imports
  `application.use_cases.AssistantService` directly.
- `graphs/*_graph.py` re-export the assistant capability graph definitions.
  New code should import from `capabilities/<feature>/graphs/`.
- `services/retriever.py` and `services/backend_client.py` retain compatibility
  names while their contracts/errors live under `application/ports` and
  `application/errors.py`.

These seams are intentionally small and removable; they are not a second
runtime path for the HTTP application.
