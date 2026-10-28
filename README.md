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
migrate application tables and bootstrap native saver.
Server and frontend use ReAct V2 only; there are no runtime enable/version flags.
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
| Structured logs / run traces | infrastructure/observability.py |
| Offline dataset/export/evaluation | training/; repository data in training/pilot-v1/ |
| Qwen model-call adapter | infrastructure/providers/qwen.py; vLLM runs separately |

Local log store: `docker compose -f ../backend/docker-compose.yml up -d lgtm`.
Set `AI_OTLP_ENDPOINT=http://localhost:4318`, then open
[AI Runs](http://localhost:3300/d/pcshop-ai) and filter by the run's `trace_id`.
[Observability guide](../docs/04-ai/observability.md) covers fields and limits.

## Verify

Training plan/pilot (not a trained model): [guide](../docs/04-ai/training-plan.md)
and [scenario index](training/pilot-v1/SCENARIOS.md). Validate offline without GPU:

```bash
uv run python -m ai_service.training training/pilot-v1
```

All pilot labels are DRAFT; validation is not human approval or proof of model quality.
Approved-label export and captured-response scoring: [offline workflow](training/README.md).
These commands do not run inference/training or replace the runtime model.
Qwen3-8B SFT/QLoRA, real inference commands and opt-in serving: [runbook](../docs/04-ai/training.md).
Model weights/GPU packages are not loaded in the FastAPI process.

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

## GitHub Actions

The existing workflow verifies every push, pull request, merge queue (`merge_group`)
and manual dispatch. It installs locked dependencies, runs Ruff/mypy/core pytest,
then applies Alembic and native saver bootstrap to a disposable PostgreSQL18 database
before running `integration_tests`. No live model/auth/catalog credentials are needed.

New CI runs cancel older runs of the same workflow on the same branch or PR,
including release branches.

Only pushes/manual dispatches on `release/vX.Y.Z` (or `release/vX.Y.Z-rc.N`)
build/publish an image tagged `X.Y.Z` (or `X.Y.Z-rc.N`) and `sha-<full commit>`.
For example, `release/v1.0.0` publishes `ai-service:1.0.0`. Other branches/tags, PRs and merge queues
verify only. Select the published AI tag in infra's tracked `release.toml`,
then run the infra release workflow to deploy. Infra stops the old
executor and runs migrations before starting AI. Server env comes from
`infra/.env.docker`, never from an env
file baked into the image. Configure SSH only in infra; see `../infra/README.md`.
Select `Verify Python service` as a required branch check; no branch rules are
modified by this workflow. Migration failures require operator action, not an
automatic schema rollback.
