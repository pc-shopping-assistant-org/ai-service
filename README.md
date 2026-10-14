# AI service

FastAPI + Pydantic + LangGraph service for catalog-grounded chat and PC consultation.
Application responses use exactly {data, message, errors}; message is a static key.

## Documentation

Design specifications live in the workspace docs repository, not duplicated here.
If cloning ai-service alone, clone docs beside it (workspace ai-service/ and docs/).

- [AI documentation index](../docs/README.md#đọc-ai-service)
- [Architecture](../docs/02-architecture/ai-architecture.md)
- [ReAct/tools](../docs/04-ai/agent-tools.md)
- [State/recovery](../docs/04-ai/context-management.md)
- [Optimizer/policies](../docs/04-ai/pc-build-optimizer.md)
- [API contract](../docs/05-api/stateful-chat-v1.md)
- [Implementation status/evidence](../docs/04-ai/stateful-implementation-status.md)
- [Delivery plan](../docs/04-ai/stateful-chat-implementation-plan.md)

Legacy chat/search/consult/compare/evaluate are request-scoped. Stateful conversation
create/get and guided/ReAct execution are opt-in; recovery and SSE remain unfinished.
Create now persists an empty accepted native root before returning READY at revision
zero. Run `uv run alembic upgrade head` (0005_event_payload) before enabling it.
Source helpers/tests do not prove complete runtime. Status/evidence has dated gates.

## Run locally

From ai-service/, copy .env.example to ignored .env only if it does not exist.
Do not overwrite local credentials. Settings load AI_ variables from .env;
shell environment overrides dotenv.

```bash
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv sync
uv run uvicorn ai_service.main:app --reload
```

Swagger: /docs; ReDoc: /redoc; OpenAPI: /openapi.json.

AI_BACKEND_API_URL is the catalog API root; the client appends /products.
Use host/Compose values from .env.example rather than copying deployment secrets.
AI_PROVIDER=fallback is local deterministic mode; openai/gemini require the matching
AI_OPENAI_API_KEY/AI_GEMINI_API_KEY. AI_MODEL_NAME is a bare model ID override.
Provider:model inference and built-in test models are removed.

## Local PostgreSQL and stateful foundation

Use the existing backend Compose postgres/ai_db, not a second database:

```bash
docker compose -f ../backend/docker-compose.yml up -d postgres
# Needed only if an older initialized volume does not yet contain ai_db.
docker compose -f ../backend/docker-compose.yml exec -T \
  -e POSTGRES_MULTIPLE_DATABASES=ai_db postgres \
  bash /docker-entrypoint-initdb.d/init-multiple-databases.sh
```

Set AI_DATABASE_URL and explicit local-only AI_TEST_POSTGRES_DSN in ignored .env
using the configured host port/credentials. Never use production for integration tests.

```bash
uv run alembic upgrade head &&
uv run python -m ai_service.bootstrap_stateful &&
uv run pytest -q integration_tests
```

Application tables are Alembic-owned; native saver tables framework-owned.
No HTTP-request migrations, direct saver edits, volume reset or silent RAM fallback.
Enable foundation routes with AI_STATEFUL_ENABLED=true and JWT JWKS/issuer/audience
settings from .env.example. Startup validates configuration/migration/saver readiness;
This enables conversation/history reads. For the guided-build submit/get run
executor, also set AI_STATEFUL_BUILD_ENABLED=true, an explicit model provider/key,
and AI_HARDWARE_CATALOG_MAPPING using real category/option UUIDs (see
[mapping contract](../docs/04-ai/agent-tools.md)). Startup validates mapping; model
clients are reused and closed after executor drain. Defaults keep execution off.
SKU clarification accepts typed or model-interpreted free-text choices, with
canonical validation before whole-draft merge. Set AI_STATEFUL_REACT_ENABLED=true
instead of AI_STATEFUL_BUILD_ENABLED for extraction → merge → trusted task
resolution → bounded ReAct → publish on the same thread (default version `v1`). Catalog read turns skip
hardware scans; build tasks still require deterministic optimizer outcomes.
Native SQL tests cover build then catalog read through lifespan/HTTP/executor,
using explicit model/auth/catalog fixtures, not live provider proof. Both execution
flags cannot be enabled together. Saved-build explanation uses the authorized
accepted snapshot (historical spending, explicit stale label), not current prices.
For the flexible V2 runtime, also select `AI_STATEFUL_REACT_VERSION=v2`.
The same routes then require explicit `schema_version: 2` and return the allowlisted
V2 accepted view; no implicit migration of accepted V1 execution checkpoints.
V2 loads the bounded canonical hardware snapshot before its flexible loop.
V2 may retry one allowlisted SDK failure after the owning executor has drained
and persisted stop ACK. It resumes the exact working ref and retained catalog;
request retries do not spawn executions. Lease-only/process-death takeover is off.
See [V2 request/view contract](../docs/05-api/stateful-chat-v2.md).
Arbitrary older-run selectors, task clarification continuation, curated web facts,
SSE, unit clarification and full setup remain open.
Guided runtime requires the backend's new `/products/snapshot` endpoint; see the
[snapshot contract](../docs/05-api/catalog-snapshot.md). An older backend fails
explicitly instead of falling back to a partial/N+1 catalog.

## Catalog indexing (optional)

Canonical backend retrieval is default. hybrid/qdrant requires Qdrant and embedding
configuration. hash embeddings are development-only; production embedding selection
is a separate decision. Index only active catalog and revalidate backend visibility.

```bash
AI_RETRIEVAL_BACKEND=hybrid \
AI_QDRANT_URL=http://localhost:6333 \
AI_EMBEDDING_PROVIDER=hash \
uv run ai-index-catalog
```

[Retrieval/freshness contract](../docs/04-ai/rag.md).

## Verification

```bash
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv run pytest -q integration_tests
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv run ruff check src tests integration_tests migrations
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv run mypy src
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv lock --check
```

Record exact environment/command/result in the central status/tracker; do not infer
PostgreSQL persistence or live-provider behavior from in-memory/unit tests.
