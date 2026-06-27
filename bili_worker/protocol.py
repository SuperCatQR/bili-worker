"""NDJSON stdio transport codec for the worker protocol (contract §4).

Pure protocol layer — **no** ``bilibili_api``. Defines the request/response envelopes
and the single-line compact JSON framing both ends share. The main process has the
mirror of this; keeping the wire shape in one reviewed module avoids drift.

Frame rule (§4.1): every frame is one line of compact UTF-8 JSON
(``json.dumps(..., ensure_ascii=False, separators=(",", ":"))``) terminated by ``\n``;
payload newlines are escaped by ``json.dumps``, so a frame never contains a bare ``\n``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Status = Literal["ok", "error"]


class ProtocolError(ValueError):
    """A malformed / non-conforming frame (contract §4.2, §11 exception state).

    Raised on decode of an unparseable or structurally invalid frame. The caller
    surfaces it explicitly (as a ``protocol_error`` error pack) rather than silently
    swallowing or misclassifying it as a business failure.
    """


def encode_frame(obj: dict[str, Any]) -> str:
    """Serialize one envelope to a single-line NDJSON frame (trailing ``\\n``)."""
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    # Invariant guard: compact dumps must never emit an embedded newline.
    assert "\n" not in line, "compact JSON frame must be single-line"
    return line + "\n"


def decode_frame(line: str) -> dict[str, Any]:
    """Parse one NDJSON frame to a dict. Raises :class:`ProtocolError` if malformed."""
    line = line.strip()
    if not line:
        raise ProtocolError("empty frame")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"frame must be a JSON object, got {type(obj).__name__}")
    return obj


@dataclass(frozen=True)
class Request:
    """Request envelope: ``{"id": int, "op": str, "params": object}`` (§4.1)."""

    id: int
    op: str
    params: dict[str, Any]

    def to_frame(self) -> str:
        return encode_frame({"id": self.id, "op": self.op, "params": self.params})

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> Request:
        try:
            req_id = obj["id"]
            op = obj["op"]
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"request missing id/op: {obj!r}") from exc
        if not isinstance(req_id, int) or isinstance(req_id, bool):
            raise ProtocolError(f"request id must be int: {obj!r}")
        if not isinstance(op, str):
            raise ProtocolError(f"request op must be str: {obj!r}")
        params = obj.get("params", {})
        if not isinstance(params, dict):
            raise ProtocolError(f"request params must be object: {obj!r}")
        return cls(id=req_id, op=op, params=params)


def ok_response(req_id: int, data: dict[str, Any]) -> dict[str, Any]:
    """Build a success response envelope (§4.2)."""
    return {"id": req_id, "status": "ok", "data": data}


def error_response(req_id: int, error: dict[str, Any]) -> dict[str, Any]:
    """Build an error response envelope carrying an error pack (§4.2, §7.1)."""
    return {"id": req_id, "status": "error", "error": error}


@dataclass(frozen=True)
class Response:
    """Decoded response: exactly one of ``data`` / ``error`` per ``status`` (§4.2)."""

    id: int
    status: Status
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> Response:
        try:
            resp_id = obj["id"]
            status = obj["status"]
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"response missing id/status: {obj!r}") from exc
        if not isinstance(resp_id, int) or isinstance(resp_id, bool):
            raise ProtocolError(f"response id must be int: {obj!r}")
        if status == "ok":
            data = obj.get("data")
            if not isinstance(data, dict):
                raise ProtocolError(f"ok response must carry data object: {obj!r}")
            return cls(id=resp_id, status="ok", data=data)
        if status == "error":
            error = obj.get("error")
            if not isinstance(error, dict):
                raise ProtocolError(f"error response must carry error object: {obj!r}")
            return cls(id=resp_id, status="error", error=error)
        raise ProtocolError(f"response status must be ok|error: {obj!r}")


__all__ = [
    "ProtocolError",
    "Request",
    "Response",
    "Status",
    "decode_frame",
    "encode_frame",
    "error_response",
    "ok_response",
]
