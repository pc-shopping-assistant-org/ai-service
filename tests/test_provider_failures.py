import httpx
import pytest
from google.genai.errors import APIError
from openai import APIStatusError

from ai_service.capabilities.assistant.recovery_policy import RetryFailure
from ai_service.infrastructure.providers.failures import provider_failure


@pytest.mark.parametrize(
    "code,expected",
    [
        (429, RetryFailure.UPSTREAM_RATE_LIMIT),
        (503, RetryFailure.UPSTREAM_SERVER_ERROR),
        (401, None),
        (400, None),
    ],
)
@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_only_allowlisted_provider_statuses_are_retryable(provider, code, expected):
    error = (
        APIStatusError(
            "fixture",
            response=httpx.Response(
                code, request=httpx.Request("POST", "https://model.test/chat")
            ),
            body=None,
        )
        if provider == "openai"
        else APIError(code, {"error": {"message": "fixture"}})
    )
    assert provider_failure(error) == expected


def test_application_timeout_is_not_assumed_to_be_a_provider_timeout():
    assert provider_failure(TimeoutError("application timeout")) is None


@pytest.mark.parametrize(
    "code,expected",
    [
        (429, RetryFailure.UPSTREAM_RATE_LIMIT),
        (503, RetryFailure.UPSTREAM_SERVER_ERROR),
        (401, None),
        (400, None),
    ],
)
def test_qwen_http_statuses_use_existing_recovery_policy(code, expected):
    response = httpx.Response(
        code, request=httpx.Request("POST", "http://qwen.test/v1/chat/completions")
    )
    assert (
        provider_failure(
            httpx.HTTPStatusError(
                "fixture", request=response.request, response=response
            )
        )
        == expected
    )


def test_qwen_timeout_uses_existing_recovery_policy():
    assert (
        provider_failure(httpx.ReadTimeout("fixture")) == RetryFailure.UPSTREAM_TIMEOUT
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        (429, RetryFailure.UPSTREAM_RATE_LIMIT),
        (503, RetryFailure.UPSTREAM_SERVER_ERROR),
        (504, RetryFailure.UPSTREAM_TIMEOUT),
        (400, None),
        (401, None),
    ],
)
def test_langchain_gemini_wrappers_preserve_status_classification(code, expected):
    from langchain_google_genai.chat_models import (
        GoogleAPIError,
        GoogleAuthenticationError,
        GoogleInvalidRequestError,
        GoogleRateLimitError,
    )

    payload = {"error": {"message": "fixture"}}
    if code >= 500:
        wrapped = GoogleAPIError(code, payload)
    else:
        wrapped = {
            400: GoogleInvalidRequestError,
            401: GoogleAuthenticationError,
            429: GoogleRateLimitError,
        }[code]("fixture")
        wrapped.__cause__ = APIError(code, payload)
    assert provider_failure(wrapped) == expected
