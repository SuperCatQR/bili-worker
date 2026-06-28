"""bili_worker entry point — stdio NDJSON event loop (contract §4–§5).

This is the **only** process that imports ``bilibili_api``. The main ``bili_unit``
process spawns it as a subprocess and talks the arm's-length stdio JSON protocol
defined in ``docs/ipc-contract-f2.md``.

The loop reads one-line compact JSON frames from stdin, dispatches to the
appropriate op handler, and writes one-line compact JSON response frames to
stdout. All logging goes to stderr so it never contaminates the protocol stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from bili_worker import PROTOCOL_VERSION, __version__
from bili_worker.catalog import ENDPOINT_BY_NAME, catalog_manifest
from bili_worker.credential import (
    CredentialError,
    EnvFileError,
    MissingCredentialsError,
    credential_get,
    credential_open,
    credential_resolve,
    load_env,
)
from bili_worker.errors import protocol_error_pack
from bili_worker.protocol import (
    ProtocolError,
    Request,
    decode_frame,
    encode_frame,
    error_response,
    ok_response,
)
from bili_worker.sdk_adapter import (
    fetch_endpoint,
    fetch_item,
    init_http_backend,
    login_qr_check,
    login_qr_generate,
    map_internal_exception,
    resolve_audio_url,
)

logger = logging.getLogger("bili_worker")


# ---------------------------------------------------------------------------
# Op handlers (contract §5)
# ---------------------------------------------------------------------------

async def handle_handshake(_params: dict[str, Any]) -> dict[str, Any]:
    """Return worker identity and protocol version (contract §5.1)."""
    try:
        import bilibili_api
        sdk_version = bilibili_api.__version__
    except Exception:
        sdk_version = "unknown"
    return {
        "worker_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "bilibili_api_version": sdk_version,
    }


async def handle_shutdown(_params: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge shutdown request (contract §5.2)."""
    return {"acknowledged": True}


async def handle_ping(_params: dict[str, Any]) -> dict[str, Any]:
    """Heartbeat / liveness check (contract §5.3)."""
    return {"pong": True}


async def handle_describe_catalog(_params: dict[str, Any]) -> dict[str, Any]:
    """Return endpoint catalog manifest (contract §5.3)."""
    endpoints = catalog_manifest()
    return {
        "endpoints": endpoints,
        "protocol_version": PROTOCOL_VERSION,
        "count": len(endpoints),
    }


async def handle_init_http_backend(params: dict[str, Any]) -> dict[str, Any]:
    """Configure bilibili-api-python HTTP backend (contract §5.7)."""
    backend = params.get("backend", "aiohttp")
    impersonate = params.get("impersonate", "chrome131")
    init_http_backend(backend, impersonate)
    return {"backend": backend, "impersonate": impersonate}


async def handle_credential_open(params: dict[str, Any]) -> dict[str, Any]:
    """Load .env and register a Credential in the pool (contract §5.4)."""
    env_path = params.get("env_path")
    return credential_open(env_path=env_path)


async def handle_credential_status(_params: dict[str, Any]) -> dict[str, Any]:
    """Return the current credential pool status."""
    from bili_worker.credential import _credential_pool
    has_credential = len(_credential_pool) > 0
    refs = list(_credential_pool.keys())
    return {"has_credential": has_credential, "refs": refs}


async def handle_fetch_page(params: dict[str, Any]) -> dict[str, Any]:
    """Call one page of a uid-level endpoint (contract §5.5)."""
    uid = params["uid"]
    endpoint = params["endpoint"]
    request_params = params.get("request_params", {})
    credential_ref = params.get("credential_ref")
    timeout = params.get("timeout", 30.0)

    spec = ENDPOINT_BY_NAME.get(endpoint)
    if spec is None:
        raise ProtocolError(f"unknown endpoint: {endpoint}")
    if spec.kind == "item":
        raise ProtocolError(f"endpoint {endpoint} is item-level, use fetch_item")

    credential = credential_resolve(credential_ref)
    return await fetch_endpoint(uid, spec, credential, request_params, timeout=timeout)


async def handle_fetch_item(params: dict[str, Any]) -> dict[str, Any]:
    """Call one item-level endpoint (contract §5.6)."""
    item_id = params["item_id"]
    endpoint = params["endpoint"]
    credential_ref = params.get("credential_ref")
    extra = params.get("extra")
    timeout = params.get("timeout", 30.0)

    spec = ENDPOINT_BY_NAME.get(endpoint)
    if spec is None:
        raise ProtocolError(f"unknown endpoint: {endpoint}")
    if spec.kind != "item":
        raise ProtocolError(f"endpoint {endpoint} is uid-level, use fetch_page")

    credential = credential_resolve(credential_ref)
    return await fetch_item(item_id, spec, credential, extra, timeout=timeout)


async def handle_resolve_audio_url(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve CDN audio stream URL (contract §5.8)."""
    bvid = params["bvid"]
    credential_ref = params.get("credential_ref")
    audio_quality = params.get("audio_quality")

    credential = credential_resolve(credential_ref)
    return await resolve_audio_url(bvid, credential, audio_quality)


async def handle_login_qr_start(_params: dict[str, Any]) -> dict[str, Any]:
    """Generate QR code for login."""
    return await login_qr_generate()


async def handle_login_qr_poll(params: dict[str, Any]) -> dict[str, Any]:
    """Poll QR code login status."""
    qrcode_key = params["qrcode_key"]
    result = await login_qr_check(qrcode_key)
    return result


async def handle_login_save_env(_params: dict[str, Any]) -> dict[str, Any]:
    """Save the current credential in the pool to .env."""
    from bili_worker.credential import _credential_pool
    if not _credential_pool:
        raise ProtocolError("no credential in pool to save")
    ref = next(iter(_credential_pool))
    cred = _credential_pool[ref]
    env_path = Path.cwd() / ".env"
    written = False
    if os.path.exists(env_path):
        # Back up existing .env
        backup = env_path.with_suffix(".env.bak")
        env_path.rename(backup)
    try:
        lines = [
            f'BILI_SESSDATA={cred.sessdata or ""}',
            f'BILI_JCT={cred.bili_jct or ""}',
            f'BILI_BUVID3={cred.buvid3 or ""}',
            f'BILI_BUVID4={getattr(cred, "buvid4", "") or ""}',
            f'BILI_DEDEUSERID={getattr(cred, "dedeuserid", "") or ""}',
            f'BILI_AC_TIME_VALUE={getattr(cred, "ac_time_value", "") or ""}',
        ]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written = True
    except OSError as exc:
        raise ProtocolError(f"failed to write .env: {exc}") from exc
    return {"saved": written, "path": str(env_path)}


async def handle_unknown_op(op: str, _params: dict[str, Any]) -> dict[str, Any]:
    """Return a protocol_error for unknown ops (contract §4.2)."""
    raise ProtocolError(f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Op dispatch table
# ---------------------------------------------------------------------------

_OP_TABLE: dict[str, Any] = {
    "handshake": handle_handshake,
    "shutdown": handle_shutdown,
    "ping": handle_ping,
    "describe_catalog": handle_describe_catalog,
    "init_http_backend": handle_init_http_backend,
    "credential_open": handle_credential_open,
    "credential_status": handle_credential_status,
    "fetch_page": handle_fetch_page,
    "fetch_item": handle_fetch_item,
    "resolve_audio_url": handle_resolve_audio_url,
    "login_qr_start": handle_login_qr_start,
    "login_qr_poll": handle_login_qr_poll,
    "login_save_env": handle_login_save_env,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def dispatch(req: Request) -> dict[str, Any]:
    """Route a request to its handler and return the response envelope."""
    handler = _OP_TABLE.get(req.op, handle_unknown_op)
    try:
        data = await handler(req.params)
        return ok_response(req.id, data)
    except ProtocolError as exc:
        return error_response(req.id, protocol_error_pack(str(exc)))
    except (CredentialError, MissingCredentialsError, EnvFileError) as exc:
        from bili_worker.errors import credential_error_pack
        return error_response(req.id, credential_error_pack(exc))
    except Exception as exc:
        error_pack = map_internal_exception(req.op, exc)
        return error_response(req.id, error_pack)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_loop() -> None:
    """Read stdin line-by-line, dispatch, write stdout line-by-line."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            line_str = line.decode("utf-8")
            try:
                obj = decode_frame(line_str)
                req = Request.from_obj(obj)
            except ProtocolError as exc:
                err_frame = encode_frame(
                    {"id": -1, "status": "error", "error": protocol_error_pack(str(exc))}
                )
                sys.stdout.write(err_frame)
                sys.stdout.flush()
                continue

            resp = await dispatch(req)
            sys.stdout.write(encode_frame(resp))
            sys.stdout.flush()

            if req.op == "shutdown":
                break
    finally:
        transport.close()


def main() -> None:
    """Entry point registered as ``bili-worker`` in ``[project.scripts]``."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_loop())


if __name__ == "__main__":
    main()
