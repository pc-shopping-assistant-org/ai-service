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
architecture: capability use cases depend on application ports, PydanticAI
and Pydantic Graph live behind infrastructure adapters, and FastAPI resolves
the application port from the composition root. Add new vertical slices under
`src/ai_service/capabilities/<feature>/`.

## Run locally

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

When `AI_MODEL_NAME` is set using PydanticAI's `provider:model` format while
`AI_PROVIDER=fallback`, the legacy model selection remains supported. Chat,
consult, compare and evaluate use the lazy `ShoppingAnswer` adapter. Provider
or network failure falls back to the deterministic answer so local integration
remains available. The streaming chat route uses a text-only adapter and emits
the same fallback as one delta when a provider is not configured. Semantic/vector search is opt-in through the
`AI_RETRIEVAL_BACKEND` setting. Internally, search/chat/consult depend on the
`CatalogRetriever` protocol and use `BackendCatalogRetriever` by default.
`hybrid` or `qdrant` additionally require `AI_QDRANT_URL` and either an HTTP
embedding endpoint or the explicit local `hash` provider. The same three routes
first run the deterministic `shopping_graph` to normalize the query and select
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
