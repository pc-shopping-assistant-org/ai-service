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


@pytest.fixture
def test_models_config_path(tmp_path) -> str:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
default_model: test-local
models:
  - id: test-local
    label: Test Local
    provider: openai
    model_name: gpt-4o-mini
    base_url: http://127.0.0.1:11434/v1
    api_key_optional: true
""",
        encoding="utf-8",
    )
    return str(config)
