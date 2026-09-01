"""Graph orchestration port.

Use cases depend on a small ``run`` contract instead of importing the
Pydantic Graph runtime. This makes graph execution replaceable in tests and
leaves room for durable/background graph runners later.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


class GraphRunner(Protocol[InputT_contra, OutputT_co]):
    """Execute one validated graph input and return its output."""

    async def run(self, inputs: InputT_contra) -> OutputT_co:
        """Run the graph with a fresh state for this request."""


__all__ = ["GraphRunner"]
