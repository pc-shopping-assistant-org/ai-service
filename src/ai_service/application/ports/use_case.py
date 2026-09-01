"""Generic inbound use-case contract for future capabilities."""

from __future__ import annotations

from typing import Protocol, TypeVar

RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
ResponseT_co = TypeVar("ResponseT_co", covariant=True)


class UseCase(Protocol[RequestT_contra, ResponseT_co]):
    """One application operation independent of HTTP or a model vendor."""

    async def execute(self, request: RequestT_contra) -> ResponseT_co:
        """Execute the capability request."""


__all__ = ["UseCase"]
