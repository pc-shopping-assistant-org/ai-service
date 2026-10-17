"""Selected model identity; update revision deliberately, never follow main at run time."""

from typing import Final

QWEN_MODEL_ID: Final = "Qwen/Qwen3-8B"
QWEN_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN_ADAPTER_NAME = "gearpc-qwen3-8b"
QWEN_MAX_CONTEXT = 32768
QWEN_REQUEST_TIMEOUT_SECONDS = 120
