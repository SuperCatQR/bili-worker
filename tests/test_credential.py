"""Tests for worker-side credential self-loading (CHO-50).

Covers the three acceptance scenarios:
  - Normal: .env present, BILI_SESSDATA set → Credential constructed
  - Empty: .env missing or SESSDATA empty → MissingCredentialsError
  - Error: .env unreadable → EnvFileError

Plus: pool lifecycle, credential_resolve, sanitize_message, env-var priority.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bili_worker.credential import (
    EnvFileError,
    MissingCredentialsError,
    build_credential,
    credential_get,
    credential_open,
    credential_pool_clear,
    credential_resolve,
    load_env,
    sanitize_message,
)


@pytest.fixture(autouse=True)
def _clean_pool_and_env():
    """Reset credential pool and clear BILI_* env vars before each test."""
    credential_pool_clear()
    for key in (
        "BILI_SESSDATA", "BILI_JCT", "BILI_BUVID3", "BILI_BUVID4",
        "BILI_DEDEUSERID", "BILI_AC_TIME_VALUE", "BILI_ENV_FILE",
    ):
        os.environ.pop(key, None)
    yield
    credential_pool_clear()
    for key in (
        "BILI_SESSDATA", "BILI_JCT", "BILI_BUVID3", "BILI_BUVID4",
        "BILI_DEDEUSERID", "BILI_AC_TIME_VALUE", "BILI_ENV_FILE",
    ):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Scenario 1: Normal — .env present, credential loads
# ---------------------------------------------------------------------------


def test_credential_open_normal(tmp_path: Path):
    """Worker reads .env, constructs Credential, returns ref."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BILI_SESSDATA=test_sessdata\nBILI_JCT=test_jct\n", encoding="utf-8",
    )

    result = credential_open(env_path=env_file)
    assert result["credential_ref"].startswith("cred-")
    assert result["has_sessdata"] is True

    cred = credential_get(result["credential_ref"])
    assert cred.sessdata == "test_sessdata"
    assert cred.bili_jct == "test_jct"


def test_credential_open_with_all_fields(tmp_path: Path):
    """All six BILI_* fields are loaded into Credential."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BILI_SESSDATA=s\nBILI_JCT=j\nBILI_BUVID3=b3\n"
        "BILI_BUVID4=b4\nBILI_DEDEUSERID=d\nBILI_AC_TIME_VALUE=a\n",
        encoding="utf-8",
    )
    result = credential_open(env_path=env_file)
    cred = credential_get(result["credential_ref"])
    assert cred.sessdata == "s"
    assert cred.bili_jct == "j"
    assert cred.buvid3 == "b3"
    assert cred.buvid4 == "b4"
    assert cred.dedeuserid == "d"
    assert cred.ac_time_value == "a"


def test_credential_open_sessdata_only(tmp_path: Path):
    """Minimal credential: only BILI_SESSDATA set."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_SESSDATA=minimal\n", encoding="utf-8")
    result = credential_open(env_path=env_file)
    cred = credential_get(result["credential_ref"])
    assert cred.sessdata == "minimal"
    assert cred.bili_jct in ("", None)


# ---------------------------------------------------------------------------
# Scenario 2: Empty — .env missing or SESSDATA empty
# ---------------------------------------------------------------------------


def test_credential_open_missing_env_file():
    """No .env file and no env vars → MissingCredentialsError."""
    with pytest.raises(MissingCredentialsError, match="BILI_SESSDATA"):
        credential_open(env_path="/nonexistent/path/.env")


def test_credential_open_empty_sessdata(tmp_path: Path):
    """.env exists but BILI_SESSDATA is empty → MissingCredentialsError."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_SESSDATA=\nBILI_JCT=something\n", encoding="utf-8")
    with pytest.raises(MissingCredentialsError, match="BILI_SESSDATA"):
        credential_open(env_path=env_file)


def test_credential_open_no_sessdata_key(tmp_path: Path):
    """.env exists but has no BILI_SESSDATA key at all."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_JCT=only_jct\n", encoding="utf-8")
    with pytest.raises(MissingCredentialsError, match="BILI_SESSDATA"):
        credential_open(env_path=env_file)


# ---------------------------------------------------------------------------
# Scenario 3: Error — .env unreadable or malformed
# ---------------------------------------------------------------------------


def test_credential_open_unreadable_env(tmp_path: Path):
    """.env exists but is a directory (unreadable as file) → EnvFileError."""
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    with pytest.raises(EnvFileError):
        credential_open(env_path=env_dir)


# ---------------------------------------------------------------------------
# Env var priority: env var overrides .env (python-dotenv override=False)
# ---------------------------------------------------------------------------


def test_env_var_overrides_dotenv(tmp_path: Path, monkeypatch):
    """When BILI_SESSDATA is already in os.environ, it takes priority."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_SESSDATA=from_file\n", encoding="utf-8")
    monkeypatch.setenv("BILI_SESSDATA", "from_env")

    result = credential_open(env_path=env_file)
    cred = credential_get(result["credential_ref"])
    assert cred.sessdata == "from_env"


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


def test_credential_pool_multiple_refs(tmp_path: Path):
    """Multiple credential_open calls produce distinct refs."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_SESSDATA=a\n", encoding="utf-8")

    r1 = credential_open(env_path=env_file)
    r2 = credential_open(env_path=env_file)
    assert r1["credential_ref"] != r2["credential_ref"]
    assert credential_get(r1["credential_ref"]).sessdata == "a"
    assert credential_get(r2["credential_ref"]).sessdata == "a"


def test_credential_get_missing_ref():
    """Accessing an invalid ref raises KeyError."""
    with pytest.raises(KeyError, match="credential_ref"):
        credential_get("cred-999")


def test_credential_resolve_null():
    """credential_resolve(None) returns None (anonymous)."""
    assert credential_resolve(None) is None


def test_credential_resolve_valid(tmp_path: Path):
    """credential_resolve with valid ref returns Credential."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_SESSDATA=x\n", encoding="utf-8")
    result = credential_open(env_path=env_file)
    cred = credential_resolve(result["credential_ref"])
    assert cred is not None
    assert cred.sessdata == "x"


def test_credential_resolve_invalid():
    """credential_resolve with invalid ref raises KeyError."""
    with pytest.raises(KeyError):
        credential_resolve("cred-nonexistent")


# ---------------------------------------------------------------------------
# build_credential standalone
# ---------------------------------------------------------------------------


def test_build_credential_from_dict():
    """build_credential works with an explicit dict."""
    cred = build_credential({"BILI_SESSDATA": "s", "BILI_JCT": "j"})
    assert cred.sessdata == "s"
    assert cred.bili_jct == "j"


def test_build_credential_missing_sessdata():
    """build_credential raises when SESSDATA missing from dict."""
    with pytest.raises(MissingCredentialsError):
        build_credential({"BILI_JCT": "j"})


# ---------------------------------------------------------------------------
# load_env standalone
# ---------------------------------------------------------------------------


def test_load_env_returns_dict(tmp_path: Path):
    """load_env returns a dict of BILI_* values."""
    env_file = tmp_path / ".env"
    env_file.write_text("BILI_SESSDATA=s\nBILI_JCT=j\n", encoding="utf-8")
    values = load_env(env_path=env_file)
    assert values["BILI_SESSDATA"] == "s"
    assert values["BILI_JCT"] == "j"
    assert values["BILI_BUVID3"] == ""


def test_load_env_missing_file_no_error():
    """load_env does not raise when .env is missing — returns empty strings."""
    values = load_env(env_path="/nonexistent/.env")
    assert values["BILI_SESSDATA"] == ""


# ---------------------------------------------------------------------------
# sanitize_message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "SESSDATA=abc123",
    "bili_jct=xyz",
    "BUVid3=foo",
    "dedeuserid=bar",
    "ac_time_value=baz",
])
def test_sanitize_redacts_credential_keywords(message: str):
    """Messages containing credential keywords are redacted."""
    assert sanitize_message(message) == "[REDACTED credential]"


def test_sanitize_passes_normal_message():
    """Normal messages pass through unchanged."""
    assert sanitize_message("fetch_page: timeout") == "fetch_page: timeout"
    assert sanitize_message("videos: 412") == "videos: 412"


# ---------------------------------------------------------------------------
# BILI_ENV_FILE env var
# ---------------------------------------------------------------------------


def test_bili_env_file_env_var(tmp_path: Path, monkeypatch):
    """BILI_ENV_FILE env var overrides default .env path."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("BILI_SESSDATA=custom_path\n", encoding="utf-8")
    monkeypatch.setenv("BILI_ENV_FILE", str(env_file))

    result = credential_open()
    cred = credential_get(result["credential_ref"])
    assert cred.sessdata == "custom_path"
