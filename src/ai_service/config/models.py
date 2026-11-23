"""YAML-backed multi-model registry with per-model thinking mode definitions.

Secrets (API keys and private endpoint URLs) stay in environment variables
referenced via ``api_key_env`` / ``base_url_env`` and are never serialized to
public API responses.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from ai_service.config.enums import ThinkingMode, ThinkingStrategy

logger = logging.getLogger(__name__)

ModelProviderName = Literal["gemini", "openai", "qwen"]


def _coerce_yaml_thinking_value(value: object) -> object:
    """YAML 1.1 parses unquoted 'off'/'on' as booleans; normalize them safely."""
    if value is False:
        return ThinkingMode.OFF.value
    if value is True:
        return ThinkingMode.ON.value
    return value


class ModelSelectionError(ValueError):
    """Raised when a requested model_id or thinking_mode is not allowed."""

    def __init__(self, field: Literal["model_id", "thinking_mode"], message: str):
        super().__init__(message)
        self.field = field


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ThinkingStrategy = ThinkingStrategy.NONE
    supported_modes: list[ThinkingMode] = Field(default_factory=list, max_length=6)
    default_mode: ThinkingMode | None = None

    @field_validator("default_mode", mode="before")
    @classmethod
    def coerce_default_mode(cls, value: object) -> object:
        return _coerce_yaml_thinking_value(value)

    @field_validator("supported_modes", mode="before")
    @classmethod
    def coerce_supported_modes(cls, value: object) -> object:
        if isinstance(value, list):
            return [_coerce_yaml_thinking_value(item) for item in value]
        return value

    @model_validator(mode="after")
    def validate_modes(self) -> ThinkingConfig:
        if len(set(self.supported_modes)) != len(self.supported_modes):
            raise ValueError("thinking.supported_modes must contain distinct values")
        if self.strategy == ThinkingStrategy.NONE:
            if self.supported_modes or self.default_mode is not None:
                raise ValueError(
                    "thinking.supported_modes/default_mode require a non-none strategy"
                )
            return self
        if not self.supported_modes:
            raise ValueError(
                "thinking.supported_modes must not be empty when thinking.strategy is set"
            )
        if self.default_mode is None:
            self.default_mode = self.supported_modes[0]
        elif self.default_mode not in self.supported_modes:
            raise ValueError(
                "thinking.default_mode must be included in thinking.supported_modes"
            )
        return self


class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,63}$")
    label: str = Field(min_length=1, max_length=120)
    provider: ModelProviderName
    model_name: str = Field(min_length=1, max_length=200)
    base_url: str | None = Field(default=None, max_length=500)
    base_url_env: str | None = Field(default=None, max_length=120)
    api_key_env: str | None = Field(default=None, max_length=120)
    api_key_optional: bool = False
    max_output_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)


class ModelsConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: str = Field(min_length=1, max_length=64)
    models: list[ModelDefinition] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_models(self) -> ModelsConfigFile:
        ids = [entry.id for entry in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError("Model IDs in YAML configuration must be unique")
        if self.default_model not in ids:
            raise ValueError(
                f"default_model '{self.default_model}' is not declared in models"
            )
        return self


class ResolvedModelEntry(BaseModel):
    """Runtime model definition with resolved credentials (internal only)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    provider: ModelProviderName
    model_name: str
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_key_optional: bool = False
    max_output_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)


class PublicModelInfo(BaseModel):
    """Public UI-safe projection of an enabled model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    provider: ModelProviderName
    supported_thinking_modes: list[ThinkingMode] = Field(default_factory=list)
    default_thinking_mode: ThinkingMode | None = None
    is_default: bool = False


class PublicModelsCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model_id: str
    models: list[PublicModelInfo] = Field(min_length=1)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {'"', "'"} and value[-1] == value[0] and len(value) >= 2:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key and value:
            values[key] = value
    return values


def _merged_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    if environ is not None:
        return {k: v for k, v in environ.items() if v.strip()}
    merged = _read_dotenv(Path(".env"))
    for key, value in os.environ.items():
        if value.strip():
            merged[key] = value.strip()
    return merged


def _resolve_entry(
    definition: ModelDefinition, env: Mapping[str, str]
) -> ResolvedModelEntry | None:
    base_url: str | None = definition.base_url.strip() if definition.base_url else None
    if definition.base_url_env:
        env_url = env.get(definition.base_url_env, "").strip()
        if env_url:
            base_url = env_url
        elif not base_url:
            logger.warning(
                "Skipping model '%s': environment variable '%s' is not set",
                definition.id,
                definition.base_url_env,
            )
            return None

    if definition.provider == "qwen" and not base_url:
        logger.warning(
            "Skipping Qwen model '%s': base_url or base_url_env is required",
            definition.id,
        )
        return None

    api_key_raw: str | None = None
    if definition.api_key_env:
        api_key_raw = env.get(definition.api_key_env, "").strip() or None
        if api_key_raw is None and not definition.api_key_optional:
            logger.warning(
                "Skipping model '%s': environment variable '%s' is not set",
                definition.id,
                definition.api_key_env,
            )
            return None
    elif definition.provider in {"gemini", "openai"} and not definition.api_key_optional:
        logger.warning(
            "Skipping model '%s': api_key_env is required unless api_key_optional is true",
            definition.id,
        )
        return None

    return ResolvedModelEntry(
        id=definition.id,
        label=definition.label,
        provider=definition.provider,
        model_name=definition.model_name,
        base_url=base_url,
        api_key=SecretStr(api_key_raw) if api_key_raw else None,
        api_key_optional=definition.api_key_optional,
        max_output_tokens=definition.max_output_tokens,
        timeout_seconds=definition.timeout_seconds,
        thinking=definition.thinking,
    )


class ModelCatalog:
    """Validated in-memory registry of active models loaded from YAML."""

    def __init__(
        self,
        entries: list[ResolvedModelEntry],
        default_model_id: str,
    ) -> None:
        if not entries:
            raise ValueError(
                "No enabled models available from AI_MODELS_CONFIG_PATH; "
                "configure at least one model with valid credentials"
            )
        self._entries: dict[str, ResolvedModelEntry] = {
            entry.id: entry for entry in entries
        }
        self._default_model_id = (
            default_model_id
            if default_model_id in self._entries
            else entries[0].id
        )

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    @property
    def entries(self) -> tuple[ResolvedModelEntry, ...]:
        return tuple(self._entries.values())

    def public_catalog(self) -> PublicModelsCatalog:
        return PublicModelsCatalog(
            default_model_id=self._default_model_id,
            models=[
                PublicModelInfo(
                    id=entry.id,
                    label=entry.label,
                    provider=entry.provider,
                    supported_thinking_modes=list(entry.thinking.supported_modes),
                    default_thinking_mode=entry.thinking.default_mode,
                    is_default=entry.id == self._default_model_id,
                )
                for entry in self._entries.values()
            ],
        )

    def resolve(
        self,
        model_id: str | None = None,
        thinking_mode: ThinkingMode | None = None,
    ) -> tuple[ResolvedModelEntry, ThinkingMode | None]:
        target_id = model_id or self._default_model_id
        entry = self._entries.get(target_id)
        if entry is None:
            raise ModelSelectionError(
                "model_id", f"Unknown or unavailable model_id '{target_id}'"
            )
        supported = entry.thinking.supported_modes
        if thinking_mode is None:
            return entry, entry.thinking.default_mode
        if not supported:
            if thinking_mode == ThinkingMode.OFF:
                return entry, None
            raise ModelSelectionError(
                "thinking_mode",
                f"Model '{entry.id}' does not support configurable thinking modes",
            )
        if thinking_mode not in supported:
            allowed = ", ".join(mode.value for mode in supported)
            raise ModelSelectionError(
                "thinking_mode",
                f"Model '{entry.id}' does not support thinking_mode '{thinking_mode.value}' "
                f"(supported: {allowed})",
            )
        return entry, thinking_mode


def load_model_catalog(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelCatalog:
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(
            f"AI_MODELS_CONFIG_PATH does not point to an existing file: {config_path}"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Model configuration YAML must be a mapping at top level")
    parsed = ModelsConfigFile.model_validate(raw)
    env = _merged_environ(environ)
    resolved = [
        entry
        for definition in parsed.models
        if (entry := _resolve_entry(definition, env)) is not None
    ]
    return ModelCatalog(resolved, parsed.default_model)
