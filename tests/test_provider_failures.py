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
