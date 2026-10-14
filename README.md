# gearPC AI service

FastAPI/Pydantic/LangGraph service for stateful product chat and deterministic PC consultation.
PostgreSQL stores conversation/run metadata and native checkpoints.

**Start here:** [AI documentation](../docs/04-ai/README.md) →
[Architecture](../docs/02-architecture/ai-architecture.md) →
[First steps](../docs/04-ai/getting-started.md).

Clone the docs repository beside ai-service. Specs, guides and references live there;
source READMEs only provide setup/navigation.

## Chạy local

Follow [First steps](../docs/04-ai/getting-started.md) to configure DB/JWT/model/catalog,
select ReAct V2, migrate application tables and bootstrap native saver.
Frontend is V2-only; server defaults remain stateful-off/V1 until explicitly configured.
Missing dependencies fail clearly, not silent legacy/RAM fallback.

After prerequisites:

```bash
UV_CACHE_DIR=/tmp/pc-shopping-uv-cache uv sync
uv run uvicorn ai_service.main:app --reload
```

Swagger /docs; OpenAPI /openapi.json. Keep ignored .env credentials private.
[Configuration](../docs/04-ai/configuration.md) · [Troubleshooting](../docs/04-ai/troubleshooting.md).

## Source map

Root package: src/ai_service/.

| Area | Entry point |
| --- | --- |
| HTTP/SSE | api/stateful_conversations.py, api/sse.py |
| Application ports/use cases | application/ |
| Agent/tool/requirement rules | capabilities/assistant/ |
| Optimizer | capabilities/pc_builder/ |
| V2 graph | infrastructure/graph/flexible.py |
| Managed runtime wiring | infrastructure/stateful.py, infrastructure/execution/build_runtime.py |

## Verify

```bash
uv run pytest -q
uv run pytest -q integration_tests
uv run ruff check src tests integration_tests migrations
uv run mypy src
uv lock --check
```

Integration DSN must target local/test DB. [Testing](../docs/04-ai/testing.md) separates
core/native/live proof. [Status](../docs/04-ai/stateful-implementation-status.md) records
dated gates/backlog; documentation does not claim live provider/catalog readiness.
