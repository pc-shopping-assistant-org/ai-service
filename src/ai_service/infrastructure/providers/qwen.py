"""Opt-in Qwen/vLLM adapter. No GPU imports, cloud fallback or SDK retries."""

from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from pydantic import BaseModel

from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.config.enums import AIProvider
from ai_service.config.qwen import QWEN_MODEL_ID, QWEN_REQUEST_TIMEOUT_SECONDS
from ai_service.config.settings import Settings
from ai_service.infrastructure.providers.qwen_protocol import prepare_qwen_request

ADAPTER_STRUCTURED_SCHEMAS = frozenset({RequirementDraft, TaskDraftBatch})


class QwenChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = QWEN_REQUEST_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._owns_client = client is None
        self.http = client if client is not None else httpx.AsyncClient(timeout=timeout)
        self.api_key = api_key

    async def complete(
        self,
        request: dict[str, Any],
        *,
        model: str,
        decoding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            **prepare_qwen_request(request),
            **(decoding or {}),
            "model": model,
            "chat_template_kwargs": {"enable_thinking": False},
            "parallel_tool_calls": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self.http.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("QWEN_OUTPUT_TRUNCATED")
        if choice.get("finish_reason") not in {"stop", "tool_calls"}:
            raise ValueError("QWEN_RESPONSE_NOT_COMPLETED")
        return result

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()


class QwenChatModel:
    """Implements the two model-call protocols used by existing native adapters."""

    def __init__(
        self,
        client: QwenChatClient,
        model_name: str,
        max_tokens: int,
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: type[BaseModel] | None = None,
    ):
        self.client, self.model_name, self.max_tokens = client, model_name, max_tokens
        self.tools, self.schema = tools, schema

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> "QwenChatModel":
        return QwenChatModel(self.client, self.model_name, self.max_tokens, tools=tools)

    def with_structured_output(
        self, schema: type[BaseModel], **kwargs: Any
    ) -> "QwenChatModel":
        # Only trained heads use the adapter. Auxiliary choice/amendment heads
        # keep the original base weights until they have their own quality gate.
        model = (
            self.model_name
            if schema in ADAPTER_STRUCTURED_SCHEMAS
            else QWEN_MODEL_ID
        )
        return QwenChatModel(self.client, model, self.max_tokens, schema=schema)

    async def ainvoke(self, input: Any) -> Any:
        request: dict[str, Any] = {
            "messages": convert_to_openai_messages(convert_to_messages(input))
        }
        if self.tools:
            request["tools"] = [
                tool
                if tool.get("type") == "function"
                else {"type": "function", "function": tool}
                for tool in self.tools
            ]
        if self.schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema.__name__,
                    "schema": self.schema.model_json_schema(),
                },
            }
        result = await self.client.complete(
            request,
            model=self.model_name,
            decoding={"max_tokens": self.max_tokens, "temperature": 0},
        )
        message = result["choices"][0]["message"]
        message["content"] = message.get("content") or ""
        if self.schema is not None:
            if message.get("tool_calls"):
                raise ValueError("QWEN_STRUCTURED_RESPONSE_CALLED_TOOL")
            return self.schema.model_validate_json(message["content"])
        converted = convert_to_messages([message])[0]
        if not isinstance(converted, AIMessage):
            raise TypeError("QWEN_ASSISTANT_RESPONSE_REQUIRED")
        return converted


class QwenModelProvider:
    name = AIProvider.QWEN

    def __init__(self, settings: Settings, *, client: QwenChatClient | None = None):
        if settings.qwen_base_url is None:
            raise ValueError("AI_QWEN_BASE_URL is required")
        self.settings, self._client = settings, client
        self._model: QwenChatModel | None = None
        self._closed = False

    def create_model(self) -> QwenChatModel:
        if self._closed:
            raise RuntimeError("Model provider is closed")
        if self._model is None:
            if self._client is None:
                self._client = QwenChatClient(
                    str(self.settings.qwen_base_url),
                    api_key=self.settings.qwen_api_key.get_secret_value()
                    if self.settings.qwen_api_key
                    else None,
                    timeout=self.settings.qwen_request_timeout_seconds,
                )
            self._model = QwenChatModel(
                self._client,
                self.settings.qwen_model_name,
                self.settings.qwen_max_tokens,
            )
        return self._model

    async def aclose(self) -> None:
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
