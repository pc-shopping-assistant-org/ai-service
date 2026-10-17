"""Small reproducibility helpers shared by training, inference and serving commands."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.training.qwen_config import QwenConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_provenance() -> dict[str, str]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    files: dict[str, JsonValue] = {
        str(path.relative_to(PROJECT_ROOT)): file_hash(path)
        for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
    }
    files["training/serving-requirements.txt"] = file_hash(
        PROJECT_ROOT / "training/serving-requirements.txt"
    )
    return {
        "revision": revision,
        "source_hash": canonical_hash(files),
        "dependency_lock_hash": file_hash(PROJECT_ROOT / "uv.lock"),
    }


def adapter_manifest(directory: Path, config: QwenConfig) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_bytes())
    saved = QwenConfig.model_validate(manifest["config"])
    if (saved.model_id, saved.model_revision) != (
        config.model_id,
        config.model_revision,
    ):
        raise ValueError("ADAPTER_BASE_MODEL_MISMATCH")
    for name, digest in manifest["adapter_files"].items():
        path = directory / "adapter" / name
        if (
            path.resolve().parent != (directory / "adapter").resolve()
            or file_hash(path) != digest
        ):
            raise ValueError("ADAPTER_FILE_HASH_MISMATCH")
    if (
        not manifest["adapter_files"]
        or "adapter_config.json" not in manifest["adapter_files"]
        or "adapter_model.safetensors" not in manifest["adapter_files"]
    ):
        raise ValueError("ADAPTER_ARTIFACT_INCOMPLETE")
    return manifest
