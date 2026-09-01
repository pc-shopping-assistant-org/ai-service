"""Compatibility exports for the answer-generator application port.

The concrete PydanticAI implementation lives under infrastructure. Existing
tests and integrations may continue importing it from this module while new
code should depend on ``application.ports.answer_generator``.
"""

from ai_service.application.ports.answer_generator import (
    AnswerGenerator,
    StreamingAnswerGenerator,
)
from ai_service.infrastructure.providers.pydantic_ai_answer_generator import (
    PydanticAIAnswerGenerator,
)

__all__ = [
    "AnswerGenerator",
    "PydanticAIAnswerGenerator",
    "StreamingAnswerGenerator",
]
