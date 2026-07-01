"""bili_worker entry point — stdio NDJSON event loop (contract §4–§6).

This is the **only** process that imports ``bilibili_api``. The main ``bili_unit``
process spawns it as a subprocess and talks the arm's-length stdio JSON protocol
defined in ``docs/ipc-contract-f2.md``.

The loop reads one-line compact JSON frames from stdin, dispatches to the
appropriate op handler, and writes one-line compact JSON response frames to
stdout. All logging goes to stderr so it never contaminates the protocol stream.

Stage 2 Step 3+: full op dispatch with catalog, credential pool, auth, and
audio URL resolution.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any

from bili_worker import PROTOCOL_VERSION, __version__
from bili_worker.credential import CredentialError, CredentialPool
from bili_worker.errors import (
    credential_error_pack,
    map_sdk_exception,
    protocol_error_pack,
)
from bili_worker.protocol import (
    ProtocolError,
    Request,
    decode_frame,
    encode_frame,
    error_response,
    ok_response,
)

from ._catalog import (
    CATALOG_BY_NAME,
    _call_item,
    _call_page,
    _init_http_backend,
    _resolve_audio_url,
)

logger = logging.getLogger("bili_worker")


def _bilibili_api_version() -> str:
    try:
        from importlib.metadata import version
        return version("bilibili-api-python")
    except Exception:
        return "unknown"


async def handle_handshake(_params: dict[str, Any]) -> dict[str, Any]:
    """Return worker identity, protocol version, and capabilities (contract §6.1)."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker_version": __version__,
        "worker": f"bili_worker/{__version__}",  # human-readable; clients read worker_version
        "bilibili_api_version": _bilibili_api_version(),
        "capabilities": [
            "fetch_page", "fetch_item", "resolve_audio_url",
            "login_qr", "credential_ref",
        ],
    }


async def handle_describe_catalog(_params: dict[str, Any]) -> dict[str, Any]:
    """Return serializable endpoint manifest (contract §6.2).

    Main process validates ``count==63 && uid_level==33 && item_level==30``.
    """
    endpoints: list[dict[str, Any]] = []
    for ep in CATALOG_BY_NAME.values():
        endpoints.append({
            "name": ep.name,
            "kind": ep.kind,
            "credential_required": ep.credential_required,
            "pagination_strategy": ep.pagination_strategy,
            "rate_limit_key": ep.rate_limit_key,
            "params_strategy": ep.params_strategy,
            "item_id_path": ep.item_id_path,
            "item_id_paths": ep.item_id_paths,
            "items_path": ep.items_path,
            "source_endpoint": ep.source_endpoint,
            "needs_parent_uid": ep.needs_parent_uid,
        })
    uid_count = sum(1 for ep in CATALOG_BY_NAME.values() if ep.kind == "uid")
    item_count = sum(1 for ep in CATALOG_BY_NAME.values() if ep.kind == "item")
    return {
        "endpoints": endpoints,
        "count": len(endpoints),
        "uid_level": uid_count,
        "item_level": item_count,
    }


async def handle_init_http_backend(params: dict[str, Any]) -> dict[str, Any]:
    """Configure the SDK global HTTP backend (contract §5, §13 Q4)."""
    backend = params.get("backend", "aiohttp")
    impersonate = params.get("impersonate", "chrome131")
    _init_http_backend(backend, impersonate)
    return {"backend": backend}


async def handle_credential_open(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Load credential from .env, return ref (contract §6.5).

    Main side (CHO-101) sends ``env_path`` (optional). Missing ``BILI_SESSDATA``
    or an unreadable ``.env`` is a worker-internal auth failure →
    ``credential_error_pack`` (AuthError/permanent), NOT an SDK error.
    """
    env_path = params.get("env_path")
    ref, has_sessdata = await pool.open(env_path=env_path)
    return {"credential_ref": ref, "has_sessdata": has_sessdata}


async def handle_credential_status(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Check if a credential_ref is valid (contract §6.5, doctor preflight)."""
    ref = params.get("credential_ref")
    if not ref:
        raise ProtocolError("credential_status requires credential_ref")
    valid, detail = await pool.status(ref)
    return {"valid": valid, "detail": detail}


async def handle_fetch_page(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Execute a uid-level endpoint callable (contract §6.3 fetch_page).

    Returns only ``raw_payload`` — pagination (``is_last_page`` /
    ``next_request``) is computed main-side from local strategies (§6.3).
    """
    uid = params["uid"]
    endpoint = params["endpoint"]
    cred_ref = params.get("credential_ref")
    request_params = params.get("request_params", {})
    timeout_s = float(params.get("timeout_s", 30.0))
    credential = await pool.resolve(cred_ref)
    raw_payload = await _call_page(uid, endpoint, credential, request_params, timeout_s)
    return {"raw_payload": raw_payload}


async def handle_fetch_item(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Execute an item-level endpoint callable (contract §6.3 fetch_item).

    ``parent_uid`` line order (CHO-101 review S2): the main side nests it inside
    ``params.extra`` and only for ``needs_parent_uid`` endpoints. The worker
    unpacks it from ``extra`` here — never reads a top-level ``parent_uid`` —
    so both ends agree on the wire shape.
    """
    endpoint = params["endpoint"]
    item_id = params["item_id"]
    cred_ref = params.get("credential_ref")
    extra = params.get("extra") or {}
    parent_uid = extra.get("parent_uid")
    timeout_s = float(params.get("timeout_s", 30.0))
    credential = await pool.resolve(cred_ref)
    raw_payload = await _call_item(endpoint, item_id, credential, parent_uid, timeout_s)
    return {"raw_payload": raw_payload}


async def handle_resolve_audio_url(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Resolve CDN audio URL without downloading bytes (contract §6.4, §10).

    Main side (CHO-118) sends ``audio_quality`` as an int mirroring
    ``AudioQuality``; contract §6.4 also allows a ``quality`` string and a
    ``page_index``. Bytes are downloaded main-side via plain aiohttp (§10).
    """
    bvid = params["bvid"]
    page_index = int(params.get("page_index", 0))
    quality = params.get("audio_quality", params.get("quality"))
    cred_ref = params.get("credential_ref")
    credential = await pool.resolve(cred_ref)
    return await _resolve_audio_url(bvid, page_index, quality, credential)


async def handle_login_qr_start(_params: dict[str, Any]) -> dict[str, Any]:
    """Start QR login flow (contract §6.6)."""
    from ._auth import login_qr_start
    login_ref, qrcode_terminal = await login_qr_start()
    return {"login_ref": login_ref, "qrcode_terminal": qrcode_terminal}


async def handle_login_qr_poll(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Poll QR login state (contract §6.6)."""
    from ._auth import login_qr_poll
    login_ref = params.get("login_ref")
    if not login_ref:
        raise ProtocolError("login_qr_poll requires login_ref")
    state, cred_ref = await login_qr_poll(login_ref, pool)
    return {"state": state, "credential_ref": cred_ref}


async def handle_login_save_env(params: dict[str, Any], pool: CredentialPool) -> dict[str, Any]:
    """Save credential to .env (contract §6.6, §8 — worker holds plaintext)."""
    from ._auth import login_save_env
    cred_ref = params.get("credential_ref")
    env_path = params.get("env_path", ".env")
    cred = await pool.resolve(cred_ref)
    if cred is None:
        raise ProtocolError(f"credential_ref {cred_ref!r} not found")
    path = login_save_env(cred, env_path)
    return {"written": True, "path": str(path)}


async def handle_ping(_params: dict[str, Any]) -> dict[str, Any]:
    """Heartbeat / liveness check (contract §5.3)."""
    return {"pong": True}


async def handle_shutdown(_params: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge shutdown request (contract §5)."""
    return {"acknowledged": True}


async def handle_unknown_op(op: str, _params: dict[str, Any]) -> dict[str, Any]:
    """Return a protocol_error for unknown ops (contract §4.2)."""
    raise ProtocolError(f"unknown op: {op!r}")


class _Shutdown(Exception):
    """Internal signal to stop the dispatch loop carrying the final response frame."""

    def __init__(self, resp: dict[str, Any]) -> None:
        super().__init__(resp)
        self.resp = resp


async def dispatch(req: Request, pool: CredentialPool) -> dict[str, Any]:
    """Route a request to its handler and return the response envelope.

    Returns a dict suitable for ``encode_frame`` — either an ok envelope
    (``{"id": N, "status": "ok", "data": {...}}``) or an error envelope
    (``{"id": N, "status": "error", "error": {...}}``).

    Error routing (contract §7):
      - ``ProtocolError`` → ``protocol_error_pack`` (permanent).
      - ``CredentialError`` (worker-internal .env/auth) → ``credential_error_pack``
        (AuthError/permanent) — distinct from SDK errors so the main side
        rebuilds an AuthError, not a retryable RequestError.
      - any other ``Exception`` → ``map_sdk_exception`` (3-state, §7.2).
    """
    op = req.op
    req_id = req.id
    params = req.params

    try:
        if op == "handshake":
            data = await handle_handshake(params)
        elif op == "describe_catalog":
            data = await handle_describe_catalog(params)
        elif op == "init_http_backend":
            data = await handle_init_http_backend(params)
        elif op == "credential_open":
            data = await handle_credential_open(params, pool)
        elif op == "credential_status":
            data = await handle_credential_status(params, pool)
        elif op == "fetch_page":
            data = await handle_fetch_page(params, pool)
        elif op == "fetch_item":
            data = await handle_fetch_item(params, pool)
        elif op == "resolve_audio_url":
            data = await handle_resolve_audio_url(params, pool)
        elif op == "login_qr_start":
            data = await handle_login_qr_start(params)
        elif op == "login_qr_poll":
            data = await handle_login_qr_poll(params, pool)
        elif op == "login_save_env":
            data = await handle_login_save_env(params, pool)
        elif op == "ping":
            data = await handle_ping(params)
        elif op == "shutdown":
            data = await handle_shutdown(params)
            raise _Shutdown(ok_response(req_id, data))
        else:
            await handle_unknown_op(op, params)
            return error_response(req_id, protocol_error_pack(f"unknown op: {op!r}"))

        return ok_response(req_id, data)

    except _Shutdown as sd:
        return sd.resp
    except ProtocolError as exc:
        return error_response(req_id, protocol_error_pack(str(exc)))
    except CredentialError as exc:
        return error_response(req_id, credential_error_pack(exc))
    except Exception as exc:
        return error_response(req_id, map_sdk_exception(op, exc))


async def run_loop() -> None:
    """Read stdin line-by-line, dispatch, write stdout line-by-line.

    The loop exits cleanly when stdin is closed (EOF) or after processing a
    ``shutdown`` request. A malformed frame produces an error response for
    that frame but does **not** terminate the loop (contract §4.2).

    Stdin is read via ``loop.run_in_executor(None, sys.stdin.readline)`` — a
    blocking ``readline`` in the default thread executor — because
    ``loop.connect_read_pipe`` raises ``OSError: [WinError 6] 句柄无效`` under
    the Windows Proactor event loop used for subprocess pipes. This
    executor-based read works reliably on Windows, Linux, and macOS.
    """
    pool = CredentialPool()
    loop = asyncio.get_running_loop()

    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                # stdin EOF — clean exit
                break

            try:
                obj = decode_frame(line)
                req = Request.from_obj(obj)
            except ProtocolError as exc:
                # Malformed frame — respond with error, keep listening
                err_frame = encode_frame(
                    {"id": -1, "status": "error", "error": protocol_error_pack(str(exc))}
                )
                sys.stdout.write(err_frame)
                sys.stdout.flush()
                continue

            resp = await dispatch(req, pool)
            sys.stdout.write(encode_frame(resp))
            sys.stdout.flush()

            if req.op == "shutdown":
                break
    finally:
        pass


def main() -> None:
    """Entry point registered as ``bili-worker`` in ``[project.scripts]``."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_loop())


if __name__ == "__main__":
    main()
