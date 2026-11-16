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
_EXCLUDED_PROPERTIES = {
    "task_index",
    "text",
}


def model_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    def project(schema: Any) -> Any:
        if isinstance(schema, list):
            return [project(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        projected = {
            key: {
                name: project(item)
                for name, item in value.items()
                if name not in _EXCLUDED_PROPERTIES
            }
            if key == "properties"
            else project(value)
            for key, value in schema.items()
            if key not in _ANNOTATIONS
        }
        if "required" in projected and isinstance(projected["required"], list):
            projected["required"] = [
                field
                for field in projected["required"]
                if field not in _EXCLUDED_PROPERTIES
            ]
        if "anyOf" in projected:
            variants = projected["anyOf"]
            non_null = [
                v
                for v in variants
                if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            if len(variants) == 2 and len(non_null) == 1 and isinstance(non_null[0], dict):
                desc = projected.get("description")
                target = dict(non_null[0])
                del projected["anyOf"]
                projected.update(target)
                if desc and "description" not in projected:
                    projected["description"] = desc
            elif variants and all(
                isinstance(v, dict) and v.get("type") == "string" and "enum" in v
                for v in variants
            ):
                merged_enum = []
                for v in variants:
                    for e in v["enum"]:
                        if e not in merged_enum:
                            merged_enum.append(e)
                desc = projected.get("description")
                del projected["anyOf"]
                projected["type"] = "string"
                projected["enum"] = merged_enum
                if desc:
                    projected["description"] = desc
        return projected

    return project(dereference_refs(model.model_json_schema()))
