# Assistant

| Package/file | Trách nhiệm |
| --- | --- |
| `requirements/` | Canonical requirements; resolve/merge/compiler/verifier rules |
| `react/` | Bốn actions, 11 tools, typed outcomes, grounding và snapshot publication rules |
| `stateful_contracts.py`, `public_contracts.py` | State/checkpoint và public DTO contracts |
| `provenance.py`, `limits.py` | Evidence identity và feature limits |
| `graphs/` | Request-scoped planning cũ; không phải V2 runtime |

Graph/executor/DB adapters nằm trong `infrastructure/`. Agent chỉ đề xuất;
application quyết định merge và CAS publish. Accepted snapshot không đổi theo
working state trước khi publish thành công.

Spec: [architecture](../../../../../docs/02-architecture/ai-architecture.md),
[ReAct/tools](../../../../../docs/04-ai/agent-tools.md),
[state/recovery](../../../../../docs/04-ai/context-management.md).
Verification/backlog: [status](../../../../../docs/04-ai/stateful-implementation-status.md).
