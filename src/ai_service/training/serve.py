"""Print an exact vLLM launch command; only --execute starts serving/downloads."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from ai_service.config.qwen import QWEN_ADAPTER_NAME
from ai_service.training.artifacts import adapter_manifest
from ai_service.training.qwen_config import QwenConfig


def serving_command(
    config: QwenConfig,
    *,
    adapter: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8001,
    executable: str = "vllm",
) -> list[str]:
    command = [
        executable,
        "serve",
        config.model_id,
        "--revision",
        config.model_revision,
        "--tokenizer-revision",
        config.model_revision,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(config.max_length),
        "--generation-config",
        "vllm",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
    ]
    if adapter is not None:
        manifest = adapter_manifest(adapter, config)
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(manifest["config"]["lora_rank"]),
                "--lora-modules",
                f"{QWEN_ADAPTER_NAME}={adapter / 'adapter'}",
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--executable", default="vllm")
    args = parser.parse_args()
    try:
        command = serving_command(
            QwenConfig.model_validate_json(args.config.read_bytes()),
            adapter=args.adapter,
            host=args.host,
            port=args.port,
            executable=args.executable,
        )
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SERVING_FAILED: {error}\n")


if __name__ == "__main__":
    main()
