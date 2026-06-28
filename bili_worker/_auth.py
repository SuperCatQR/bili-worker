"""Worker-side QR login handlers (contract §6.6).

QR login is split into request-response polling because the worker has no TTY.
The main process calls login_qr_start → login_qr_poll (repeatedly) → login_save_env.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bilibili_api import Credential, login_v2

logger = logging.getLogger("bili_worker.auth")

# In-memory login session store: login_ref → QrCodeLogin instance.
_sessions: dict[str, login_v2.QrCodeLogin] = {}
_next_login_ref = 0


def _next_ref() -> str:
    global _next_login_ref
    _next_login_ref += 1
    return f"qr-{_next_login_ref}"


async def login_qr_start() -> tuple[str, str]:
    """Generate a QR code and return (login_ref, qrcode_terminal)."""
    qr = login_v2.QrCodeLogin(platform=login_v2.QrCodeLoginChannel.WEB)
    await qr.generate_qrcode()
    ref = _next_ref()
    _sessions[ref] = qr
    terminal_qr = qr.get_qrcode_terminal()
    return ref, terminal_qr


async def login_qr_poll(login_ref: str, pool: Any) -> tuple[str, str | None]:
    """Poll QR login state. Returns (state, credential_ref | None).

    state is one of: SCAN, CONF, TIMEOUT, DONE.
    credential_ref is set only when DONE.
    """
    qr = _sessions.get(login_ref)
    if qr is None:
        raise ValueError(f"unknown login_ref: {login_ref!r}")

    if qr.has_done():
        cred = qr.get_credential()
        # Store in credential pool
        if hasattr(pool, 'add'):
            cred_ref = pool.add(cred)
        else:
            # Fallback: use credential_open-style ref
            from bili_worker.credential import _credential_pool, _next_credential_ref
            cred_ref = _next_credential_ref()
            _credential_pool[cred_ref] = cred
        del _sessions[login_ref]
        return "DONE", cred_ref

    state = await qr.check_state()
    if state == login_v2.QrCodeLoginEvents.TIMEOUT:
        del _sessions[login_ref]
        return "TIMEOUT", None
    elif state == login_v2.QrCodeLoginEvents.SCAN:
        return "SCAN", None
    elif state == login_v2.QrCodeLoginEvents.CONF:
        return "CONF", None

    return "SCAN", None


def login_save_env(cred: Credential, env_path: str | Path = ".env") -> Path:
    """Write credential fields to .env file (contract §6.6)."""
    env_path = Path(env_path)

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    fields = {
        "BILI_SESSDATA": cred.sessdata or "",
        "BILI_JCT": cred.bili_jct or "",
        "BILI_BUVID3": cred.buvid3 or "",
        "BILI_BUVID4": cred.buvid4 or "",
        "BILI_DEDEUSERID": cred.dedeuserid or "",
        "BILI_AC_TIME_VALUE": cred.ac_time_value or "",
    }

    new_lines = [
        line for line in existing_lines
        if not any(line.startswith(f"{k}=") for k in fields)
    ]
    for key, value in fields.items():
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info("Credential saved to %s", env_path)
    return env_path


__all__ = ["login_qr_start", "login_qr_poll", "login_save_env"]
