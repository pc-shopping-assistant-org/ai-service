import asyncio
from types import SimpleNamespace

import pytest

from ai_service.config.settings import Settings
from ai_service.infrastructure.providers.model_answer_generator import (
    ModelAnswerGenerator,
)


class FakeModel:
    def __init__(self, output=None, chunks=(), error=None):
        self.output = output
        self.chunks = chunks
        self.error = error
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.output

    async def astream(self, messages):
        self.messages = messages
        for content in self.chunks:
            yield SimpleNamespace(content=content)
        if self.error:
            raise self.error


def generator_with(model):
    generator = ModelAnswerGenerator(Settings(model_system_prompt="grounded only"))
    generator._model = model
    generator._initialization_attempted = True
    return generator


@pytest.mark.asyncio
async def test_typed_output_and_system_prompt():
    model = FakeModel(output={"answer": "  grounded answer  "})
    assert (
        await generator_with(model).generate("catalog", "fallback") == "grounded answer"
    )
    assert model.schema.__name__ == "ShoppingAnswer"
    assert model.messages == [("system", "grounded only"), ("human", "catalog")]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [{"answer": 123}, {}, {"answer": " "}])
async def test_invalid_or_empty_output_uses_fallback(output):
    assert (
        await generator_with(FakeModel(output=output)).generate("catalog", "fallback")
        == "fallback"
    )


@pytest.mark.asyncio
async def test_stream_extracts_only_text_not_reasoning_or_tools():
    model = FakeModel(
        chunks=(
            "hello ",
            [
                {"type": "text", "text": "world"},
                {"type": "reasoning", "text": "private"},
            ],
        )
    )
    assert [
        part async for part in generator_with(model).stream("catalog", "fallback")
    ] == ["hello ", "world"]


@pytest.mark.asyncio
async def test_failure_before_first_delta_uses_fallback():
    model = FakeModel(error=RuntimeError("provider failed"))
    assert [
        part async for part in generator_with(model).stream("catalog", "fallback")
    ] == ["fallback"]


@pytest.mark.asyncio
async def test_failure_after_delta_is_not_reported_as_success():
    model = FakeModel(chunks=("partial",), error=RuntimeError("provider failed"))
    with pytest.raises(RuntimeError):
        async for _ in generator_with(model).stream("catalog", "fallback"):
            pass


@pytest.mark.asyncio
async def test_cancellation_propagates():
    with pytest.raises(asyncio.CancelledError):
        async for _ in generator_with(FakeModel(error=asyncio.CancelledError())).stream(
            "catalog", "fallback"
        ):
            pass


@pytest.mark.asyncio
async def test_fallback_does_not_infer_provider_from_model_name():
    generator = ModelAnswerGenerator(
        Settings(provider="fallback", model_name="openai:gpt-4o-mini")
    )
    assert await generator.generate("catalog", "fallback") == "fallback"
    assert [part async for part in generator.stream("catalog", "fallback")] == [
        "fallback"
    ]
