"""Explicit native test database, using the same dotenv precedence as the app."""

import pytest
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresTestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="AI_", extra="ignore"
    )
    test_postgres_dsn: SecretStr | None = None


@pytest.fixture
def postgres_dsn() -> str:
    configured = PostgresTestSettings().test_postgres_dsn
    if configured is None:
        pytest.fail(
            "Set AI_TEST_POSTGRES_DSN in .env or shell to a migrated local ai_db"
        )
    value = configured.get_secret_value()
    if not value.startswith("postgresql://"):
        pytest.fail("AI_TEST_POSTGRES_DSN must use postgresql:// for native PostgreSQL")
    return value
