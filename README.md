# AI service

FastAPI boundary for the PC shopping assistant. The current MVP exposes
backend-grounded chat, natural-language catalog search, consultation,
comparison and evaluation routes while keeping the frontend envelope stable:

```json
{
  "data": {},
  "message": "AI_CHAT_COMPLETED",
  "errors": []
}
```

`message` is a static key for frontend mapping. Human-readable validation,
backend and product details belong in `errors[]`.

The reusable project foundation is documented in
[`ARCHITECTURE.md`](ARCHITECTURE.md). Runtime wiring follows clean
architecture: capability use cases depend on application ports, LangGraph
and LangChain model calls live behind infrastructure adapters, and FastAPI resolves
the application port from the composition root. Add new vertical slices under
`src/ai_service/capabilities/<feature>/`.

## Run locally

### Framework migration status (2026-10-07)

Source now targets LangGraph + Pydantic + direct LangChain OpenAI/Gemini model
adapters. PydanticAI/Pydantic Graph callers and direct dependencies were removed.
There is no second agent runtime or tool loop. Shopping/comparison use a compiled
request-scoped LangGraph; durable conversation checkpoints are NOT implemented.

Migration verification is **incomplete**: this environment cannot resolve PyPI
DNS and has no cached LangGraph/model packages. `uv.lock` still belongs to the
previous dependency set; do not deploy using that lock. On a network-enabled
environment, run `uv lock`, `uv sync`, then pytest/Ruff/mypy and `uv lock --check`.
Review the resolved dependency versions before committing the migration. Graph
runtime and provider-construction tests must pass; do not skip them. ISSUE-077
in the workspace tracker records this blocker.

Stateful work started with B0 contracts and rejection tests in
`capabilities/assistant/stateful_contracts.py`. Login-only and total-setup budget
are approved; these schemas are not yet wired into routes, auth or the optimizer.
The B1 native PostgreSQL gate is explicit and mandatory:
`uv run pytest -q integration_tests/test_accepted_head_postgres.py`, using a
disposable `AI_TEST_POSTGRES_DSN`. It is a candidate invocation experiment, not
proof of persisted conversations or a selected production adapter. See
`docs/04-ai/stateful-chat-contracts.md` in the workspace for acceptance and gaps.

### Grounding and toolkit integration status

The PC-builder, commerce and web-search toolkits are library entrypoints, **not
registered tools in live chat**. No `/api/v1/pc-builder/optimize` route exists.
Live chat continues through the canonical retriever and answer generator.

- `recommend_pc_build` requires an explicit component catalog and delegates to
  `PCBuildApplicationService`; the old hardcoded rule-engine recommendation
  raises `BackendUnavailableError` instead of producing a fictional build.
- Missing cooler socket support is `UNKNOWN`, not compatible. CPU/GPU rows
  lacking TDP are rejected without aborting other valid candidates. Synthetic
  iGPU candidates cannot bypass a locked GPU brand.
- Explanation context preserves each part's original `price`, `is_owned` and
  `spending_price`. The latter sums to the build's budget total; excluded owned
  parts cost zero in that sum. Compatibility means only the modeled checks,
  not certification of every real-world hardware constraint.
- Commerce reads require active backend rows and valid detail fields (`id`,
  `name`, `category`, `listPrice`, `inStock`, `brand`, `warrantyMonths`). Invalid,
  incomplete or unavailable data is not replaced with sample products. Nested
  variant/category mapping still needs an approved canonical contract.
- Advanced category/brand/spec filters, authenticated carts/order lookup,
  promotions, store policies and atomic build export are not integrated;
  their default adapter raises `BackendUnavailableError`. A missing detail or
  cart stays absent rather than receiving an invented UUID.
- Web-search failure returns no sources with `WEB_SEARCH_UNAVAILABLE`; an empty
  successful search has `WEB_SEARCH_NO_RESULTS`. The tool preserves that note
  and must not present unavailable search as live evidence.

Runtime toolkit dispatch, catalog mapping, ownership/auth and confirmed atomic
cart mutations remain tracked under `ISSUE-076` in the workspace tracker.
Tests use HTTP fixtures/ASGI transport; they do not verify a live backend or LLM.

```bash
cp .env.example .env
uv sync
uv run uvicorn ai_service.main:app --reload
```

Swagger is available at `/docs`, ReDoc at `/redoc`, and the OpenAPI document at
`/openapi.json`. The service uses a deterministic backend-grounded fallback
until an LLM provider/model is configured.

## Environment

All settings use the `AI_` prefix. `AI_BACKEND_API_URL` should point at the
backend API's `/api/v1` root. `AI_PROVIDER=fallback` and an empty
`AI_MODEL_NAME` are the safe local defaults; no provider call is made in that
mode. Set `AI_PROVIDER=openai` or `AI_PROVIDER=gemini` to use the built-in
lazy provider adapters, then inject `AI_OPENAI_API_KEY` or `AI_GEMINI_API_KEY`
through the runtime secret store. `AI_MODEL_NAME` overrides the provider's
default model (`gpt-4o-mini` or `gemini-2.5-flash`).

Legacy `provider:model` inference and the built-in `test` model are removed.
Choose `AI_PROVIDER` explicitly and set `AI_MODEL_NAME` to the provider's bare
model ID. `AI_PROVIDER=fallback` never calls a model, even with a model name.
Chat, consult, compare and evaluate use the lazy `ShoppingAnswer` adapter. Provider
or network failure falls back to the deterministic answer so local integration
remains available. The streaming chat route uses a text-only adapter and emits
the same fallback as one delta when a provider is not configured, returns an
ERROR event after a partial provider failure, and propagates cancellation.
SDK retries are disabled; provider requests use `AI_REQUEST_TIMEOUT_SECONDS`.
Semantic/vector search is opt-in through the
`AI_RETRIEVAL_BACKEND` setting. Internally, search/chat/consult depend on the
`CatalogRetriever` protocol and use `BackendCatalogRetriever` by default.
`hybrid` or `qdrant` additionally require `AI_QDRANT_URL` and either an HTTP
embedding endpoint or the explicit local `hash` provider. The same three routes
first run the deterministic shopping planning graph to normalize the query and select
the search/consult branch. The graph owns planning only. The backend fallback
tries the full phrase first and then a small de-duplicated set of meaningful
terms when the phrase has no keyword hits; this remains lexical expansion. The
Qdrant path uses the configured embedding provider and keeps canonical product
payloads in the vector collection; run the catalog indexer before enabling
strict `qdrant` retrieval.

The idempotent `ai-index-catalog` command walks the backend catalog cursor pages
and upserts product payloads/embeddings into the configured collection:

```bash
AI_RETRIEVAL_BACKEND=hybrid \
AI_QDRANT_URL=http://localhost:6333 \
AI_EMBEDDING_PROVIDER=hash \
uv run ai-index-catalog
```

Use the HTTP embedding provider for production; the local hash provider is only
for development and contract verification.
