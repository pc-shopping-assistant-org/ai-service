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

The installed LangGraph/model stack and resolver-generated lock are now verified:
223 tests pass, including graph/provider, microservice catalog and core regression
tests; Ruff, mypy (87 source files) and `uv lock --check` (140 packages) pass. This is local fixture
verification, not live provider/backend or durable recovery acceptance.

Phase P1 adds pure multi-turn rules in `conversation_core.py`: unknown defaults,
atomic patch merge, trusted provenance/locks, owned/pinned conflicts, explicit
completion/hash/version reuse and invalidation, historical build preservation,
and total-setup budget arithmetic. Nine core scenarios exercise these outcomes,
including the real optimizer service. The module is not yet wired into chat.
P2 remains BLOCKED (ISSUE-077): Postgres saver/driver/ORM packages are unavailable
and there is no reachable test PostgreSQL. Do not bypass its native checkpoint gate.

P0 contracts are complete: `stateful_contracts.py` (scope/refs/patch),
`public_contracts.py` (allowlisted API/SSE), `provenance.py` (typed internal replay
snapshots/canonical hash). Actual stateful routes, DB and replay remain pending.
Login-only and total-setup budget
were initially approved; the final V1 policy is BUILD_PC (default, core only)
versus FULL_SETUP (core plus monitor/mouse/keyboard/headset, max one/type), no
implicit budget split, owned spending zero and pinned paid. Retention is 90 days
(orphan/debug checkpoints 7 days), cascading conversation deletion and internal
non-publishing replay. These policies are not yet wired into runtime/cleanup;
owned false is now rejected by the V1 schema. See workspace contracts.
The B1 native PostgreSQL gate is explicit and mandatory:
`uv run pytest -q integration_tests/test_accepted_head_postgres.py`, using a
local `ai_db` through `AI_TEST_POSTGRES_DSN` (owner-approved). It is a candidate invocation experiment, not
proof of persisted conversations or a selected production adapter. See
`docs/04-ai/stateful-chat-contracts.md` in the workspace for acceptance and gaps.

#### Local AI PostgreSQL

Use the existing microservices compose, not an additional PostgreSQL container:

```sh
cd ../backend
docker compose up -d postgres
# Also handles a volume initialized before ai_db was added. Does not delete data.
docker compose exec -T -e POSTGRES_MULTIPLE_DATABASES=ai_db postgres \
  bash /docker-entrypoint-initdb.d/init-multiple-databases.sh
cd ../ai-service
export AI_TEST_POSTGRES_DSN='postgresql://postgres:postgres@127.0.0.1:5432/ai_db'
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv run pytest -q \
  integration_tests/test_accepted_head_postgres.py
```

The DSN uses existing local compose defaults; replace credentials if customized.
The gate bootstraps saver tables and writes random experiment threads in `ai_db`;
it does not clear the database. Use local development data, never production.
Provisioning the database alone does not implement or verify P2 persistence.

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

All settings use the `AI_` prefix. `AI_BACKEND_API_URL` points at the catalog
API root: host microservice `http://localhost:8082` (the `.env.example` value),
Compose `http://host.docker.internal:8082`, or legacy monolith
`http://localhost:8080/api/v1`. The client appends `/products`; do not add that
path to the base URL. Existing `.env` files must be adjusted explicitly rather
than overwritten. `AI_PROVIDER=fallback` and an empty
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
