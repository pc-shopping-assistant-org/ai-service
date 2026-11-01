"""Public draft text projection; native tool arguments remain private protocol."""

from collections.abc import AsyncIterator, Awaitable, Callable
from time import monotonic
from typing import Protocol

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    message_chunk_to_message,
)

TextStreamSink = Callable[[str, bool], Awaitable[None]]
STREAM_FLUSH_CHARS = 64
STREAM_FLUSH_SECONDS = 0.05
MAX_DRAFT_CHARS = 12_000


class BoundModel(Protocol):
    async def ainvoke(self, input: list[BaseMessage]) -> AIMessage: ...

    def astream(self, input: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]: ...


def public_text(message: AIMessageChunk) -> str:
    if message.tool_call_chunks:
        # LangChain parses partial native arguments. Only these final-answer
        # fields are user-facing; never forward raw JSON or requirement drafts.
        if len(message.tool_calls) != 1:
            return ""
        call = message.tool_calls[0]
        if call["name"] not in {"submit_result", "ask_clarification"}:
            return ""
        args = call["args"]
        if call["name"] == "ask_clarification":
            question = args.get("question", "")
            return question if isinstance(question, str) else ""
        results = args.get("results", [])
        if not isinstance(results, list):
            return ""
        return "\n\n".join(
            text
            for row in results
            if isinstance(row, dict)
            for text in [row.get("explanation") or row.get("text")]
            if isinstance(text, str)
        )
    if isinstance(message.content, str):
        text = message.content
    else:
        text = "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in message.content
            if isinstance(part, str) or part.get("type") == "text"
        )
    return "" if text.lstrip().startswith(("{", "[")) else text


async def stream_public_text(
    model: BoundModel, messages: list[BaseMessage], sink: TextStreamSink
) -> AIMessage:
    aggregate: AIMessageChunk | None = None
    emitted, current = "", ""
    flushed_at = monotonic()

    async def flush() -> None:
        nonlocal emitted, flushed_at
        if not current or current == emitted:
            return
        replace = not emitted or not current.startswith(emitted)
        await sink(current if replace else current[len(emitted) :], replace)
        emitted, flushed_at = current, monotonic()

    async for chunk in model.astream(messages):
        aggregate = chunk if aggregate is None else aggregate + chunk
        current = public_text(aggregate)[:MAX_DRAFT_CHARS]
        if (
            not emitted
            or len(current) - len(emitted) >= STREAM_FLUSH_CHARS
            or monotonic() - flushed_at >= STREAM_FLUSH_SECONDS
        ):
            await flush()
    await flush()
    if aggregate is None:
        raise ValueError("Model returned an empty stream")
    response = message_chunk_to_message(aggregate)
    if not isinstance(response, AIMessage):
        raise TypeError("Expected an assistant model response")
    return response
