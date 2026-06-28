"""Worker-side credential self-loading from .env (contract §8).

This module is the worker's independent credential subsystem. It reads
BILI_SESSDATA / BILI_JCT / BILI_BUVID3 / BILI_BUVID4 / BILI_DEDEUSERID /
BILI_AC_TIME_VALUE from the process environment (which ``python-dotenv``
populates from ``.env``), constructs ``bilibili_api.Credential`` objects,
and manages them in an in-memory pool keyed by opaque ``credential_ref``
strings.

The main process never sees plaintext credentials — it only holds
``credential_ref`` handles (contract §8).

Arm's-length boundary (§12): this module imports ``bilibili_api`` and
``dotenv``. The main ``bili_unit`` process never imports this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from bilibili_api import Credential

logger = logging.getLogger("bili_worker.credential")

# ---------------------------------------------------------------------------
# Public exception types
# ---------------------------------------------------------------------------


class CredentialError(Exception):
    """Base for all credential-loading failures in the worker."""


class MissingCredentialsError(CredentialError):
    """BILI_SESSDATA is empty or missing — cannot construct a Credential."""


class EnvFileError(CredentialError):
    """.env file is unreadable (permissions, encoding) or malformed."""


# ---------------------------------------------------------------------------
# Credential field names (contract §8)
# ---------------------------------------------------------------------------

_CREDENTIAL_ENV_KEYS: tuple[str, ...] = (
    "BILI_SESSDATA",
    "BILI_JCT",
    "BILI_BUVID3",
    "BILI_BUVID4",
    "BILI_DEDEUSERID",
    "BILI_AC_TIME_VALUE",
)

# ---------------------------------------------------------------------------
# In-memory credential pool (contract §8)
# ---------------------------------------------------------------------------

_credential_pool: dict[str, Credential] = {}
_next_ref: int = 0


def _next_credential_ref() -> str:
    global _next_ref
    _next_ref += 1
    return f"cred-{_next_ref}"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def _find_env_path(env_path: str | Path | None = None) -> Path:
    """Resolve the .env file path.

    Priority:
    1. Explicit ``env_path`` argument.
    2. ``BILI_ENV_FILE`` environment variable.
    3. ``.env`` in the current working directory.

    Returns the resolved absolute Path.
    """
    if env_path is not None:
        return Path(env_path).resolve()
    env_var = os.environ.get("BILI_ENV_FILE")
    if env_var:
        return Path(env_var).resolve()
    return Path.cwd() / ".env"


def load_env(*, env_path: str | Path | None = None) -> dict[str, str]:
    """Load credential-relevant env vars, populating os.environ from .env.

    Uses ``python-dotenv`` with ``override=False`` (env vars already set
    take priority over .env file values). Returns a dict of the loaded
    BILI_* credential fields (may be empty strings).

    Raises:
        EnvFileError: .env exists but is unreadable (permissions, encoding).
    """
    from dotenv import load_dotenv as _load_dotenv

    resolved = _find_env_path(env_path)

    if resolved.exists():
        if resolved.is_dir():
            raise EnvFileError(
                f".env path is a directory, not a file: {resolved}"
            )
        try:
            _load_dotenv(dotenv_path=resolved, override=False)
        except (OSError, UnicodeDecodeError) as exc:
            raise EnvFileError(
                f"Cannot read .env at {resolved}: {exc}"
            ) from exc
    else:
        logger.debug(
            "No .env file found at %s; using process environment only", resolved,
        )

    return {
        key: os.environ.get(key, "")
        for key in _CREDENTIAL_ENV_KEYS
    }


# ---------------------------------------------------------------------------
# Credential construction
# ---------------------------------------------------------------------------


def build_credential(env_values: dict[str, str] | None = None) -> Credential:
    """Construct a ``bilibili_api.Credential`` from env values.

    Args:
        env_values: Optional pre-loaded dict of BILI_* values. If None,
            reads directly from ``os.environ`` (caller should have called
            ``load_env()`` first).

    Returns:
        A ``bilibili_api.Credential`` instance.

    Raises:
        MissingCredentialsError: BILI_SESSDATA is empty or missing.
    """
    if env_values is None:
        env_values = {key: os.environ.get(key, "") for key in _CREDENTIAL_ENV_KEYS}

    sessdata = env_values.get("BILI_SESSDATA", "").strip()
    if not sessdata:
        raise MissingCredentialsError(
            "BILI_SESSDATA is empty or not set — cannot construct credential. "
            "Run `bili-unit login` to obtain credentials, "
            "or set BILI_SESSDATA in .env.",
        )

    kwargs: dict[str, str] = {"sessdata": sessdata}

    jct = env_values.get("BILI_JCT", "").strip()
    if jct:
        kwargs["bili_jct"] = jct

    for key, attr in [
        ("BILI_BUVID3", "buvid3"),
        ("BILI_BUVID4", "buvid4"),
        ("BILI_DEDEUSERID", "dedeuserid"),
        ("BILI_AC_TIME_VALUE", "ac_time_value"),
    ]:
        value = env_values.get(key, "").strip()
        if value:
            kwargs[attr] = value

    logger.debug("Credential constructed (sessdata present=%s)", bool(sessdata))
    return Credential(**kwargs)


# ---------------------------------------------------------------------------
# Credential pool (contract §8)
# ---------------------------------------------------------------------------


def credential_open(*, env_path: str | Path | None = None) -> dict[str, Any]:
    """Load .env, construct a Credential, store it in the pool.

    Returns a dict suitable for the ``credential_open`` op response data
    (contract §6.5)::

        {"credential_ref": "cred-1", "has_sessdata": true}

    Raises:
        MissingCredentialsError: BILI_SESSDATA missing.
        EnvFileError: .env unreadable.
    """
    env_values = load_env(env_path=env_path)
    cred = build_credential(env_values)
    ref = _next_credential_ref()
    _credential_pool[ref] = cred
    logger.info("Credential opened: ref=%s", ref)
    return {"credential_ref": ref, "has_sessdata": True}


def credential_get(ref: str) -> Credential:
    """Retrieve a Credential from the pool by ref.

    Raises:
        KeyError: ref not found in pool (expired after restart, contract §8).
    """
    if ref not in _credential_pool:
        raise KeyError(
            f"credential_ref {ref!r} not found in pool — "
            f"worker may have restarted; re-open with credential_open",
        )
    return _credential_pool[ref]


def credential_resolve(ref: str | None) -> Credential | None:
    """Resolve a credential_ref to a Credential, or None for anonymous.

    ``ref=None`` returns None (anonymous request, contract §8).
    ``ref`` not in pool raises KeyError.
    """
    if ref is None:
        return None
    return credential_get(ref)


def credential_pool_clear() -> None:
    """Clear all credentials from the pool (used in tests)."""
    _credential_pool.clear()
    global _next_ref
    _next_ref = 0


# ---------------------------------------------------------------------------
# Log redaction (contract §8)
# ---------------------------------------------------------------------------

# Sensitive substrings that must never appear in log messages or error packs.
_CREDENTIAL_SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "sessdata", "bili_jct", "buvid3", "buvid4", "dedeuserid", "ac_time_value",
)


def sanitize_message(message: str) -> str:
    """Return a sanitized version of a message string.

    If the message contains any known credential-sensitive substring
    (case-insensitive), returns ``"[REDACTED credential]"`` instead.
    Otherwise returns the message unchanged.
    """
    lower = message.lower()
    for keyword in _CREDENTIAL_SENSITIVE_SUBSTRINGS:
        if keyword in lower:
            return "[REDACTED credential]"
    return message


# ---------------------------------------------------------------------------
# CredentialPool — async-friendly pool wrapper (Step 3+)
# ---------------------------------------------------------------------------


class CredentialPool:
    """Async-friendly credential pool used by the op dispatch loop.

    Wraps the module-level credential functions with async-compatible
    interfaces that the ``__main__`` dispatch handlers expect.
    """

    def __init__(self) -> None:
        self._opened = False

    async def open(self, *, reload_env: bool = False) -> tuple[str, bool]:
        """Load .env, construct Credential, store in pool, return (ref, has_sessdata)."""
        if reload_env:
            credential_pool_clear()
        result = credential_open()
        return result["credential_ref"], result["has_sessdata"]

    async def status(self, ref: str) -> tuple[bool, str]:
        """Check if a credential_ref is valid."""
        try:
            cred = credential_get(ref)
            has_sessdata = bool(cred.sessdata)
            return has_sessdata, "valid" if has_sessdata else "no sessdata"
        except KeyError:
            return False, f"credential_ref {ref!r} not found"

    async def resolve(self, ref: str | None) -> Credential | None:
        """Resolve a credential_ref to a Credential, or None for anonymous."""
        return credential_resolve(ref)

    def add(self, cred: Credential) -> str:
        """Add an existing Credential to the pool, return its ref."""
        ref = _next_credential_ref()
        _credential_pool[ref] = cred
        return ref


__all__ = [
    "CredentialError",
    "CredentialPool",
    "EnvFileError",
    "MissingCredentialsError",
    "build_credential",
    "credential_get",
    "credential_open",
    "credential_pool_clear",
    "credential_resolve",
    "load_env",
    "sanitize_message",
]
