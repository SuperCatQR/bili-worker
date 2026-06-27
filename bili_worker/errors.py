"""Worker-side SDK-exception -> serializable error pack (contract §7.2).

This is the worker half of the §7 error contract. It is the faithful port of
``bili_unit.fetching._adapter_core.map_bilibili_errors``: given an SDK exception, it
produces the **same** ``(type_name, message)`` that today's mapper would raise, plus
the 3-state ``classification`` and diagnostic ``code``. The main process rebuilds the
exact fetching exception from the ``type`` string (via its own ``_TYPE_REGISTRY``) and
classifies it unchanged — proven by ``tests/test_error_mapping_parity.py``.

Arm's-length boundary (§12): this package does **not** import ``bili_unit``. The
fetching-exception **type names** are part of the frozen wire contract (§7.2), so we
emit them as string literals here; the main side owns the string -> class mapping.
"""

from __future__ import annotations

from typing import Any, Literal

from bilibili_api.exceptions import (
    ApiException,
    ArgsException,
    CredentialNoBiliJctException,
    CredentialNoSessdataException,
    NetworkException,
    ResponseCodeException,
)

Classification = Literal["retryable", "permanent", "unavailable"]

# Mirror of bili_unit.fetching._adapter_core._PERMANENT_BUSINESS_CODES.
_PERMANENT_BUSINESS_CODES: frozenset[int] = frozenset({
    -400, 22115, 22118, 53013, 53016, 88214,
})


def _pack(
    type_: str,
    classification: Classification,
    message: str,
    code: int | None,
) -> dict[str, Any]:
    return {
        "type": type_,
        "classification": classification,
        "code": code,
        "message": message,
        "retryable_hint": classification == "retryable",
    }


def map_sdk_exception(label: str, exc: BaseException) -> dict[str, Any]:
    """Map an SDK exception to an error-pack dict, mirroring ``map_bilibili_errors``.

    The branch order and message strings match the main-process mapper exactly, so the
    rebuilt fetching exception has an identical ``str()`` (contract §7.3 zero-behaviour
    change). ``label`` is the endpoint label, same as the original mapper's argument.
    """
    if isinstance(exc, TimeoutError):
        return _pack("Http5xxError", "retryable", f"{label}: timeout", None)

    if isinstance(exc, ResponseCodeException):
        if exc.code == 412:
            return _pack("Http412Error", "retryable", f"{label}: 412", 412)
        if exc.code in _PERMANENT_BUSINESS_CODES:
            return _pack(
                "ResourceUnavailableError",
                "unavailable",
                f"{label}: code={exc.code}: {exc.msg}",
                exc.code,
            )
        return _pack(
            "RequestError",
            "retryable",
            f"{label}: code={exc.code}: {exc.msg}",
            exc.code,
        )

    if isinstance(exc, NetworkException):
        status = getattr(exc, "status", 0) or 0
        if status == 404:
            return _pack(
                "ResourceUnavailableError",
                "unavailable",
                f"{label}: HTTP 404 (route gone): {exc}",
                404,
            )
        if 400 <= status < 500:
            return _pack("RequestError", "retryable", f"{label}: HTTP {status}: {exc}", status)
        return _pack("Http5xxError", "retryable", f"{label}: network error {exc}", status or None)

    if isinstance(exc, (CredentialNoSessdataException, CredentialNoBiliJctException)):
        return _pack("AuthError", "permanent", f"{label}: credential missing: {exc}", None)

    if isinstance(exc, ArgsException):
        return _pack(
            "InvalidRequestError",
            "permanent",
            f"{label}: invalid SDK arguments: {exc}",
            None,
        )

    if isinstance(exc, ApiException):
        return _pack("RequestError", "retryable", f"{label}: {exc}", None)

    return _pack("RequestError", "retryable", f"{label}: unexpected: {exc}", None)


def download_error_pack(message: str) -> dict[str, Any]:
    """Error pack for an audio-download failure (contract §6.4/§7.2).

    ``resolve_audio_url`` (Step 6) uses this for "no audio stream / CDN resolve failed".
    It is classified ``permanent`` explicitly — the main side reads this ``classification``
    / rebuilds a permanent fetching exception, so it is NOT retried. (The reviewer flagged
    that the generic codec default is ``retryable``; this helper is the explicit override.)
    """
    return _pack("ResourceUnavailableError", "permanent", message, None)


def protocol_error_pack(message: str) -> dict[str, Any]:
    """Error pack for a protocol-level fault (unknown op / bad frame / missing field, §4.2)."""
    return {
        "type": "protocol_error",
        "classification": "permanent",
        "code": None,
        "message": message,
        "retryable_hint": False,
    }


__all__ = [
    "Classification",
    "download_error_pack",
    "map_sdk_exception",
    "protocol_error_pack",
]
