import json

import httpx
import pytest
from test_training_dataset import dataset, sample

from ai_service.infrastructure.providers.qwen import QwenChatClient
from ai_service.training.artifacts import file_hash
from ai_service.training.compare import compare_reports
from ai_service.training.inference import capture_responses
from ai_service.training.qwen_config import QwenConfig
from ai_service.training.serve import serving_command


def approved(**changes):
    return sample(
        split="validation",
        review_status="APPROVED",
        reviewed_by="fixture-reviewer",
        **changes,
    )


def adapter(path, config):
    path.mkdir()
    (path / "adapter").mkdir()
    for name, content in {
        "adapter_config.json": "{}",
        "adapter_model.safetensors": "test-only-not-real-weights",
    }.items():
        (path / "adapter" / name).write_text(content)
    manifest = {
        "config": config.model_dump(),
        "adapter_files": {p.name: file_hash(p) for p in (path / "adapter").iterdir()},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.mark.asyncio
async def test_capture_evaluates_actual_response_not_gold_and_retains_failure(tmp_path):
    source = dataset(
        tmp_path / "source",
        [
            approved(sample_id="valid", scenario_id="valid"),
            approved(sample_id="error", scenario_id="error"),
        ],
    )
    calls = []

    async def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 2:
            return httpx.Response(503, json={"error": "fixture unavailable"})
        # A different action from the authored CALL_TOOL label, schema valid only.
        candidate = {
            "schema_version": 2,
            "action": {
                "kind": "ASK_CLARIFICATION",
                "question": "Hệ điều hành nào?",
                "missing_fields": ["operating_system"],
            },
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(candidate),
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        result = await capture_responses(
            source,
            tmp_path / "baseline",
            QwenConfig(),
            client=QwenChatClient("http://qwen.test/v1", client=http),
        )
    assert result["total"] == 2 and result["validated"] == 1
    assert result["cases"][0]["matches_authored_target"] is False
    assert result["cases"][1]["status"] == "FAIL"
    assert all("target" not in body and "reviewed_by" not in body for body in calls)
    assert all("fixture-reviewer" not in str(body) for body in calls)
    assert all(body["model"] == "Qwen/Qwen3-8B" for body in calls)
    assert (tmp_path / "baseline" / "responses.jsonl").exists()


def report(adapter_revision=None):
    return {
        "schema_version": 1,
        "dataset": {"files": {"samples.jsonl": "hash"}},
        "split": "validation",
        "run": {
            "model_id": "Qwen/Qwen3-8B",
            "model_revision": QwenConfig().model_revision,
            "code_revision": "code",
            "decoding": {"temperature": 0},
            "adapter_revision": adapter_revision,
            "seed": 42,
        },
        "semantic_review": "REQUIRED",
        "cases": [
            {
                "sample_id": "s1",
                "request_hash": "r1",
                "model_input_hash": "p1",
                "status": "PASS",
                "raw_contract_valid": True,
            }
        ],
        "total": 1,
        "raw_contract_valid": 1,
        "validated": 1,
    }


def test_compare_requires_paired_inputs_same_base_and_no_auto_promotion():
    before, after = report(), report("adapter-sha")
    comparison = compare_reports(before, after)
    assert comparison["validated_delta"] == 0
    assert comparison["promotion"] == "MANUAL_REVIEW_REQUIRED"
    after["cases"][0]["request_hash"] = "different"
    with pytest.raises(ValueError, match="INPUTS_DIFFER"):
        compare_reports(before, after)


@pytest.mark.parametrize("field", ["model_revision", "decoding", "seed"])
def test_compare_rejects_changed_experiment_conditions(field):
    before, after = report(), report("adapter-sha")
    after["run"][field] = "changed"
    with pytest.raises(ValueError, match="CONDITIONS_DIFFER"):
        compare_reports(before, after)


def test_serving_uses_pinned_base_and_adapter_does_not_replace_tokenizer(tmp_path):
    config = QwenConfig()
    base = serving_command(config)
    assert base[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert base[base.index("--revision") + 1] == config.model_revision
    path = adapter(tmp_path / "trained", config)
    command = serving_command(config, adapter=path)
    assert "--enable-lora" in command
    assert "--tokenizer" not in command
    assert str(path / "adapter") in command[command.index("--lora-modules") + 1]
    assert "--chat-template-kwargs" in command
    (path / "adapter" / "adapter_model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        serving_command(config, adapter=path)


def test_serving_rejects_manifest_with_no_adapter_weights(tmp_path):
    config = QwenConfig()
    path = adapter(tmp_path / "trained", config)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["adapter_files"].pop("adapter_model.safetensors")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ARTIFACT_INCOMPLETE"):
        serving_command(config, adapter=path)
