# Capabilities

Business logic theo feature; không chứa HTTP routes hoặc sở hữu provider/DB clients.

| Package | Vai trò |
| --- | --- |
| `assistant/requirements/` | Draft, resolve, atomic merge, compiler và satisfaction verifier |
| `assistant/react/` | Actions, state-aware tools, ledger, grounding và public projection |
| `pc_builder/` | Deterministic optimizer, compatibility, scoring và build application service |
| `web_search/` | Search/read contracts và query presets |
| `shopping_assistant/` | Commerce helpers; không mặc nhiên là tools đã expose |

LangGraph V2 ở `infrastructure/graph/flexible.py`; runtime wiring ở
`infrastructure/execution/build_runtime.py`. Application gọi qua ports, không
cho model truy cập trực tiếp DB hoặc tự publish state.

Thêm feature vào package sở hữu rule; không tạo global `utils.py`/`service.py`.
Tên `graphs/` trong assistant hiện phục vụ request-scoped planning cũ, không phải
stateful ReAct runtime. [Architecture](../../../../docs/02-architecture/ai-architecture.md).
