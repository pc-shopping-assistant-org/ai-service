"""Allowlisted SDK failures only; malformed output/programming bugs stay terminal."""

from collections.abc import Awaitable, Callable

import httpx

from ai_service.capabilities.assistant.recovery_policy import RetryFailure
from ai_service.infrastructure.execution.supervisor import ExecutionFailed
from ai_service.infrastructure.observability import step


def provider_failure(error: Exception) -> RetryFailure | None:
    if isinstance(error, httpx.TimeoutException):
        return RetryFailure.UPSTREAM_TIMEOUT
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 429:
            return RetryFailure.UPSTREAM_RATE_LIMIT
        if 500 <= error.response.status_code < 600:
            return RetryFailure.UPSTREAM_SERVER_ERROR
        return None
    module = type(error).__module__
    if module.startswith("openai"):
        from openai import APIStatusError, APITimeoutError

        if isinstance(error, APITimeoutError):
            return RetryFailure.UPSTREAM_TIMEOUT
        if isinstance(error, APIStatusError):
            if error.status_code == 429:
                return RetryFailure.UPSTREAM_RATE_LIMIT
            if 500 <= error.status_code < 600:
                return RetryFailure.UPSTREAM_SERVER_ERROR
    elif module.startswith("google.genai.errors"):
        from google.genai.errors import APIError

        if isinstance(error, APIError):
            if error.code == 429:
                return RetryFailure.UPSTREAM_RATE_LIMIT
            if error.code == 504:
                return RetryFailure.UPSTREAM_TIMEOUT
            if 500 <= error.code < 600:
                return RetryFailure.UPSTREAM_SERVER_ERROR
    return None


async def invoke_provider[T](dispatch: Callable[[], Awaitable[T]]) -> T:
    try:
        with step("model.call"):
            return await dispatch()
    except Exception as error:
        failure = provider_failure(error)
        if failure is None:
            raise
        raise ExecutionFailed(failure) from error
