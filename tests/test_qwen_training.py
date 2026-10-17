import json
from pathlib import Path

import pytest
from test_training_dataset import dataset, sample

from ai_service.training.qwen_config import QwenConfig
from ai_service.training.qwen_data import tokenize_example
from ai_service.training.train import train


class Tokenizer:
    """Tiny deterministic template fake, not evidence of real Qwen tokenization."""

    chat_template = "fixture-template"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        assert kwargs["enable_thinking"] is False
        text = "".join(
            f"<{message['role']}>{message.get('content', '')}{json.dumps(message['tool_calls']) if message.get('tool_calls') else ''}</end>"
            for message in messages
        )
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(text.encode()),
            "attention_mask": [1] * len(text.encode()),
        }


def test_config_fixes_model_and_requires_immutable_revision():
    config = QwenConfig()
    assert config.model_id == "Qwen/Qwen3-8B"
    assert len(config.model_revision) == 40
    with pytest.raises(ValueError):
        QwenConfig(model_revision="main")
    with pytest.raises(ValueError):
        QwenConfig(model_id="other/model")


def test_only_final_completion_has_loss_and_schema_reaches_model():
    example = {
        "request": {
            "messages": [{"role": "user", "content": "RAM"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Draft", "schema": {"type": "object"}},
            },
        },
        "target": {"role": "assistant", "content": "{}"},
    }
    row = tokenize_example(example, Tokenizer(), max_length=2048)
    assert len(row["labels"]) == len(row["input_ids"])
    supervised = bytes(token for token in row["labels"] if token != -100).decode()
    assert supervised == "{}</end>"
    prompt = bytes(
        token
        for token, label in zip(row["input_ids"], row["labels"], strict=True)
        if label == -100
    ).decode()
    assert "Draft" in prompt and "RAM" in prompt


def test_overlong_examples_fail_instead_of_silently_truncating():
    example = {
        "request": {"messages": [{"role": "user", "content": "long"}]},
        "target": {"role": "assistant", "content": "{}"},
    }
    with pytest.raises(ValueError, match="TOO_LONG"):
        tokenize_example(example, Tokenizer(), max_length=1)


def test_unreviewed_dataset_fails_before_loading_gpu_libraries(tmp_path):
    source = dataset(tmp_path / "source", [sample()])
    with pytest.raises(ValueError, match="DATASET_REVIEW_REQUIRED"):
        train(source, tmp_path / "run", QwenConfig())
    assert not (tmp_path / "run").exists()


def test_checked_in_qwen_config_is_readable():
    path = Path(__file__).parents[1] / "training" / "qwen3-8b.json"
    assert QwenConfig.model_validate_json(path.read_bytes()).model_id == "Qwen/Qwen3-8B"


def test_completed_tool_history_is_context_not_training_target():
    example = {
        "request": {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior-tool-call"},
                {"role": "tool", "content": "prior-result"},
            ]
        },
        "target": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_web", "arguments": '{"query":"GPU"}'}}
            ],
        },
    }
    row = tokenize_example(example, Tokenizer(), max_length=2048)
    supervised = bytes(token for token in row["labels"] if token != -100).decode()
    assert "search_web" in supervised
    assert "prior-tool-call" not in supervised and "prior-result" not in supervised


def test_recipe_builds_nf4_adapter_and_artifacts_without_real_gpu(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace as NS

    import ai_service.training.train as module

    samples = [
        sample(
            sample_id=f"s-{split}",
            scenario_id=f"family-{split}",
            split=split,
            review_status="APPROVED",
            reviewed_by="fixture",
        )
        for split in ("train", "validation", "test")
    ]
    source = dataset(tmp_path / "source", samples)
    observed = {}

    def options(name):
        def capture(*args, **kwargs):
            observed[name] = kwargs
            return NS(**kwargs)

        return capture

    class SavedTokenizer(Tokenizer):
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")

    class Trainer:
        def __init__(self, **kwargs):
            observed["trainer"] = kwargs

        def train(self):
            return NS(metrics={"train_loss": 0.4})

        def save_model(self, path):
            target = Path(path)
            target.mkdir()
            (target / "adapter_config.json").write_text("{}")
            (target / "adapter_model.safetensors").write_text("fixture-not-weights")

    model = NS(config=NS(use_cache=True))
    modules = {
        "torch": NS(
            cuda=NS(
                is_available=lambda: True,
                is_bf16_supported=lambda: True,
                get_device_name=lambda device: "fixture-GPU",
            ),
            bfloat16="bf16",
            float16="fp16",
        ),
        "transformers": NS(
            set_seed=lambda seed: None,
            AutoTokenizer=NS(from_pretrained=lambda *a, **kw: SavedTokenizer()),
            AutoModelForCausalLM=NS(from_pretrained=lambda *a, **kw: model),
            BitsAndBytesConfig=options("quantization"),
            DataCollatorForSeq2Seq=options("collator"),
        ),
        "datasets": NS(Dataset=NS(from_list=lambda rows: rows)),
        "peft": NS(
            prepare_model_for_kbit_training=lambda model, **kw: model,
            get_peft_model=lambda model, config: model,
            LoraConfig=options("lora"),
        ),
        "trl": NS(SFTConfig=options("sft"), SFTTrainer=Trainer),
    }
    seen = []

    def tokenized(example, tokenizer, **kwargs):
        seen.append(example["sample_id"])
        return {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}

    monkeypatch.setattr(module, "import_module", modules.__getitem__)
    monkeypatch.setattr(module, "tokenize_example", tokenized)
    output = tmp_path / "trained"
    result = module.train(source, output, QwenConfig())
    assert seen == ["s-train", "s-validation"]
    assert observed["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert observed["lora"]["target_modules"] == "all-linear"
    assert observed["sft"]["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert observed["sft"]["max_length"] is None
    assert observed["trainer"]["train_dataset"][0]["labels"] == [-100, 2]
    assert result["adapter_files"]["adapter_model.safetensors"]
    assert json.loads((output / "manifest.json").read_text())["compute_dtype"] == "bf16"
