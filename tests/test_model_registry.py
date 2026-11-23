"""Tests for YAML multi-model registry, thinking mode wiring, and /models API."""

from uuid import uuid4

import httpx
import pytest

from ai_service.capabilities.assistant.public_contracts import RunViewV1
from ai_service.capabilities.assistant.react.turn_contracts import (
    TurnRequestV2,
    turn_request_payload,
)
from ai_service.config.enums import ThinkingMode
from ai_service.config.models import ModelSelectionError, load_model_catalog
from ai_service.config.settings import Settings
from ai_service.infrastructure.providers.registry import ModelProviderRegistry
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.main import create_app

SAMPLE_YAML = """
default_model: gemini-2.5-flash
models:
  - id: gemini-2.5-flash
    label: "Gemini 2.5 Flash"
    provider: gemini
    model_name: gemini-2.5-flash
    api_key_env: GEMINI_TEST_KEY
    max_output_tokens: 4096
    timeout_seconds: 120
    thinking:
      strategy: gemini_reasoning
      supported_modes: [off, low, medium, high]
      default_mode: low

  - id: qwen3.5-ollama
    label: "Qwen 3.5 9B (Ollama)"
    provider: openai
    model_name: qwen3.5:9b-q4_K_M
    base_url: http://127.0.0.1:11434/v1
    api_key_optional: true
    max_output_tokens: 4096
    timeout_seconds: 180
    thinking:
      strategy: ollama_think
      supported_modes: [off, on]
      default_mode: on

  - id: missing-key-openai
    label: "Unavailable Model"
    provider: openai
    model_name: gpt-4o
    api_key_env: MISSING_OPENAI_KEY
"""


def test_yaml_loader_filters_missing_credentials_and_hides_secrets(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(SAMPLE_YAML, encoding="utf-8")

    catalog = load_model_catalog(cfg, environ={"GEMINI_TEST_KEY": "super-secret"})
    assert catalog.default_model_id == "gemini-2.5-flash"
    assert [m.id for m in catalog.entries] == ["gemini-2.5-flash", "qwen3.5-ollama"]

    public = catalog.public_catalog()
    dumped = public.model_dump_json()
    assert "super-secret" not in dumped
    assert "127.0.0.1:11434" not in dumped
    assert public.default_model_id == "gemini-2.5-flash"
    assert [m.id for m in public.models] == ["gemini-2.5-flash", "qwen3.5-ollama"]
    assert public.models[0].is_default is True
    assert public.models[1].supported_thinking_modes == [
        ThinkingMode.OFF,
        ThinkingMode.ON,
    ]


def test_yaml_loader_falls_back_default_to_first_enabled_and_fails_when_none(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(SAMPLE_YAML, encoding="utf-8")

    # Gemini key absent -> only local Ollama (api_key_optional=true) stays enabled.
    catalog = load_model_catalog(cfg, environ={})
    assert catalog.default_model_id == "qwen3.5-ollama"
    assert [m.id for m in catalog.entries] == ["qwen3.5-ollama"]

    only_missing = tmp_path / "empty.yaml"
    only_missing.write_text(
        """
default_model: need-key
models:
  - id: need-key
    label: Need Key
    provider: gemini
    model_name: gemini-2.5-flash
    api_key_env: MISSING_KEY
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No enabled models"):
        load_model_catalog(only_missing, environ={})


def test_catalog_resolve_validates_model_and_thinking_mode(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(SAMPLE_YAML, encoding="utf-8")
    catalog = load_model_catalog(cfg, environ={"GEMINI_TEST_KEY": "k"})

    entry, mode = catalog.resolve()
    assert entry.id == "gemini-2.5-flash"
    assert mode == ThinkingMode.LOW

    entry, mode = catalog.resolve("qwen3.5-ollama", ThinkingMode.OFF)
    assert entry.id == "qwen3.5-ollama"
    assert mode == ThinkingMode.OFF

    with pytest.raises(ModelSelectionError) as exc_model:
        catalog.resolve("nonexistent")
    assert exc_model.value.field == "model_id"

    with pytest.raises(ModelSelectionError) as exc_mode:
        catalog.resolve("qwen3.5-ollama", ThinkingMode.HIGH)
    assert exc_mode.value.field == "thinking_mode"


@pytest.mark.asyncio
async def test_provider_registry_caches_and_wires_ollama_think_flag(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(SAMPLE_YAML, encoding="utf-8")
    catalog = load_model_catalog(cfg, environ={"GEMINI_TEST_KEY": "k"})
    registry = ModelProviderRegistry(catalog)
    try:
        p_on_1, _, m_on = registry.resolve_provider("qwen3.5-ollama", ThinkingMode.ON)
        p_on_2, _, _ = registry.resolve_provider("qwen3.5-ollama", ThinkingMode.ON)
        p_off, _, m_off = registry.resolve_provider("qwen3.5-ollama", ThinkingMode.OFF)
        assert p_on_1 is p_on_2
        assert p_on_1 is not p_off
        assert m_on == ThinkingMode.ON
        assert m_off == ThinkingMode.OFF

        llm_on = p_on_1.create_model()
        llm_off = p_off.create_model()
        assert getattr(llm_on, "extra_body", None) == {"think": True}
        assert getattr(llm_off, "extra_body", None) == {"think": False}
    finally:
        await registry.aclose()


def test_turn_request_v2_round_trips_model_id_and_thinking_mode():
    req = TurnRequestV2(
        conversation_id=uuid4(),
        request_id=uuid4(),
        expected_revision=0,
        message="Tư vấn GPU",
        model_id="qwen3.5-ollama",
        thinking_mode=ThinkingMode.OFF,
    )
    payload = turn_request_payload(req)
    assert payload["model_id"] == "qwen3.5-ollama"
    assert payload["thinking_mode"] == "off"


@pytest.mark.asyncio
async def test_models_endpoint_and_run_validation(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(SAMPLE_YAML, encoding="utf-8")
    catalog = load_model_catalog(cfg, environ={"GEMINI_TEST_KEY": "secret-value"})
    owner, conv_id, req_id, run_id = (uuid4() for _ in range(4))
    submitted: list[TurnRequestV2] = []

    class Auth:
        async def authenticate(self, token: str):
            return owner

    class Runs:
        async def submit(self, account, request):
            submitted.append(request)
            return RunViewV1(
                run_id=run_id,
                conversation_id=conv_id,
                request_id=req_id,
                status="PENDING",
                base_revision=0,
            )

    app = create_app(Settings(_env_file=None))
    app.state.stateful = StatefulResources(
        conversations=object(),  # type: ignore[arg-type]
        authenticator=Auth(),  # type: ignore[arg-type]
        runs=Runs(),  # type: ignore[arg-type]
        models=catalog,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/models")
        assert res.status_code == 200
        body = res.json()
        assert body["message"] == "AI_MODELS_LISTED"
        assert body["data"]["default_model_id"] == "gemini-2.5-flash"
        assert "secret-value" not in res.text

        bad_model = await client.post(
            f"/api/v1/conversations/{conv_id}/runs",
            headers={"Authorization": "Bearer ok"},
            json={
                "schema_version": 2,
                "conversation_id": str(conv_id),
                "request_id": str(req_id),
                "expected_revision": 0,
                "message": "Hi",
                "model_id": "unknown-model",
            },
        )
        assert bad_model.status_code == 422
        assert bad_model.json()["errors"][0]["field"] == "model_id"

        bad_thinking = await client.post(
            f"/api/v1/conversations/{conv_id}/runs",
            headers={"Authorization": "Bearer ok"},
            json={
                "schema_version": 2,
                "conversation_id": str(conv_id),
                "request_id": str(req_id),
                "expected_revision": 0,
                "message": "Hi",
                "model_id": "qwen3.5-ollama",
                "thinking_mode": "high",
            },
        )
        assert bad_thinking.status_code == 422
        assert bad_thinking.json()["errors"][0]["field"] == "thinking_mode"

        ok = await client.post(
            f"/api/v1/conversations/{conv_id}/runs",
            headers={"Authorization": "Bearer ok"},
            json={
                "schema_version": 2,
                "conversation_id": str(conv_id),
                "request_id": str(req_id),
                "expected_revision": 0,
                "message": "Hi",
                "model_id": "qwen3.5-ollama",
                "thinking_mode": "off",
            },
        )
        assert ok.status_code == 202
        assert submitted[-1].model_id == "qwen3.5-ollama"
        assert submitted[-1].thinking_mode == ThinkingMode.OFF

