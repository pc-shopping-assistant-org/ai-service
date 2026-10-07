"""Compatibility exports for the answer-generator application port.

The concrete model implementation lives under infrastructure. Existing
tests and integrations may continue importing it from this module while new
code should depend on ``application.ports.answer_generator``.
"""

from ai_service.application.ports.answer_generator import (
    AnswerGenerator,
    StreamingAnswerGenerator,
)
from ai_service.infrastructure.providers.model_answer_generator import (
    ModelAnswerGenerator,
)

__all__ = [
    "AnswerGenerator",
    "ModelAnswerGenerator",
    "StreamingAnswerGenerator",
]
