import pytest


@pytest.mark.parametrize("shell_dsn", [None, "postgresql://localhost/explicit_test"])
def test_native_test_reads_dotenv_without_export_and_shell_override_wins(
    tmp_path, monkeypatch, request, shell_dsn
):
    monkeypatch.delenv("AI_TEST_POSTGRES_DSN", raising=False)
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_TEST_POSTGRES_DSN=postgresql://localhost/local_test\nAI_PROVIDER=fallback\n"
    )
    if shell_dsn:
        monkeypatch.setenv("AI_TEST_POSTGRES_DSN", shell_dsn)
    assert request.getfixturevalue("postgres_dsn") == (
        shell_dsn or "postgresql://localhost/local_test"
    )
