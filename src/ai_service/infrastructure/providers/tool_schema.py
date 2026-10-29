"""Model projection only; Pydantic validators keep their complete constraints."""

from typing import Any

from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel

_ANNOTATIONS = {
    "title",
    "default",
    "examples",
    "$comment",
    "$defs",
    "definitions",
    "additionalProperties",
    "discriminator",
}


def model_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    def project(schema: Any) -> Any:
        if isinstance(schema, list):
            return [project(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        return {
            key: {name: project(item) for name, item in value.items()}
            if key == "properties"
            else project(value)
            for key, value in schema.items()
            if key not in _ANNOTATIONS
        }

    return project(dereference_refs(model.model_json_schema()))
