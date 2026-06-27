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
import sys
from typing import Any

from bili_worker import PROTOCOL_VERSION, __version__
from bili_worker.errors import map_sdk_exception, protocol_error_pack
from bili_worker.protocol import (
    ProtocolError,
    Request,
    decode_frame,
    encode_frame,
    error_response,
    ok_response,
)

async def handle_handshake(_params: dict[str, Any]) -> dict[str, Any]:
    """Return worker identity and protocol version (contract §5.1)."""
    return {
        "worker_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
    }

async def handle_shutdown(_params: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge shutdown request (contract §5.2)."""
    return {"acknowledged": True}

async def handle_ping(_params: dict[str, Any]) -> dict[str, Any]:
    """Heartbeat / liveness check (contract §5.3)."""
    return {"pong": True}

async def handle_unknown_op(op: str, _params: dict[str, Any]) -> dict[str, Any]:
    """Return a protocol_error for unknown ops (contract §4.2)."""
    raise ProtocolError(f"unknown op: {op}")

#: Op dispatch table.  Keys are the op string from the request envelope.
#: Handlers receive the ``params`` dict and return a ``data`` dict (for ok
#: responses) or raise :class:`ProtocolError` (for protocol-level errors).
#: SDK-level errors are caught by the main loop and mapped via
#: :func:`map_sdk_exception`.
_OP_TABLE: dict[str, Any] = {
    "handshake": handle_handshake,
    "shutdown": handle_shutdown,
    "ping": handle_ping,
}

async def dispatch(req: Request) -> dict[str, Any]:
    """Route a request to its handler and return the response envelope.

    Returns a dict suitable for ``encode_frame`` — either an ok envelope
    (``{"id": N, "status": "ok", "data": {...}}``) or an error envelope
    (``{"id": N, "status": "error", "error": {...}}``).
    """
    handler = _OP_TABLE.get(req.op, handle_unknown_op)
    try:
        data = await handler(req.params)
        return ok_response(req.id, data)
    except ProtocolError as exc:
        return error_response(req.id, protocol_error_pack(str(exc)))
    except Exception as exc:
        return error_response(req.id, map_sdk_exception(req.op, exc))

async def run_loop() -> None:
    """Read stdin line-by-line, dispatch, write stdout line-by-line.

    The loop exits cleanly when stdin is closed (EOF) or after processing a
    ``shutdown`` request.  A malformed frame produces an error response for
    that frame but does **not** terminate the loop (contract §4.2).
    """
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(
        lambda: protocol, sys.stdin
    )

    try:
        while True:
            line = await reader.readline()
            if not line:
                # stdin EOF — clean exit
                break

            line_str = line.decode("utf-8")
            try:
                obj = decode_frame(line_str)
                req = Request.from_obj(obj)
            except ProtocolError as exc:
                # Malformed frame — respond with error, keep listening
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
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
