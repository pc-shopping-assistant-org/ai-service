"""Pydantic Graph adapter for the application ``GraphRunner`` port."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic_graph import Graph

from ai_service.application.ports.graph_runner import GraphRunner

StateT = TypeVar("StateT")
DepsT = TypeVar("DepsT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PydanticGraphRunner[StateT, DepsT, InputT, OutputT](GraphRunner[InputT, OutputT]):
    """Run a ``pydantic-graph`` graph with request-scoped state/dependencies.

    Graph definitions stay in the capability package. This adapter owns the
    vendor runtime call and is the only place where the application container
    needs to know about ``Graph.run(state=..., deps=..., inputs=...)``.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        state_factory: Callable[[], StateT],
        deps_factory: Callable[[], DepsT] | None = None,
    ) -> None:
        self._graph = graph
        self._state_factory = state_factory
        self._deps_factory = deps_factory

    async def run(self, inputs: InputT) -> OutputT:
        state = self._state_factory()
        if self._deps_factory is None:
            return await self._graph.run(state=state, inputs=inputs)
        return await self._graph.run(
            state=state,
            deps=self._deps_factory(),
            inputs=inputs,
        )


__all__ = ["PydanticGraphRunner"]
