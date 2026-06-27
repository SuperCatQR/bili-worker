"""Parity test: worker error mapper == main-process map_bilibili_errors (contract §7).

This is the Step 2 correctness proof. For every SDK exception the worker can hit, the
worker's :func:`bili_worker.errors.map_sdk_exception` must produce a pack whose ``type``
and ``message`` exactly match what today's ``map_bilibili_errors`` raises — so when the
main side rebuilds the exception (via ``fetching_exception_from_pack``) and classifies
it, the result is byte-identical to the pre-refactor direct path (§7.3 zero behaviour
change). The test legitimately imports both packages (dev/test only; the shipped
packages stay at arm's length).
"""

from __future__ import annotations

import pytest
from bilibili_api.exceptions import (
    ApiException,
    ArgsException,
    CredentialNoBiliJctException,
    CredentialNoSessdataException,
    NetworkException,
    ResponseCodeException,
)

from bili_unit.fetching import FetchingError
from bili_unit.fetching._adapter_core import map_bilibili_errors
from bili_unit.fetching._error_pack import ErrorPack, fetching_exception_from_pack
from bili_unit.fetching.runner._failure import classify_fetching_exception
from bili_worker.errors import download_error_pack, map_sdk_exception, protocol_error_pack
from bili_worker.protocol import decode_frame, encode_frame

_LABEL = "videos"

# Same SDK cases as bili_unit's test_fetching_error_pack, exercised through the worker.
_SDK_CASES: list[BaseException] = [
    TimeoutError(),
    ResponseCodeException(412, "too fast", {}),
    ResponseCodeException(-400, "请求错误", {}),
    ResponseCodeException(53013, "用户隐私设置未公开", {}),
    ResponseCodeException(88214, "up未开通充电", {}),
    ResponseCodeException(99999, "other", {}),
    NetworkException(404, "not found"),
    NetworkException(403, "forbidden"),
    NetworkException(500, "server"),
    NetworkException(0, "conn reset"),
    CredentialNoSessdataException(),
    CredentialNoBiliJctException(),
    ArgsException("bad input"),
    ApiException("generic api error"),
    RuntimeError("totally unexpected"),
]


async def _direct(sdk_exc: BaseException) -> FetchingError:
    """Pre-refactor path: SDK exc -> map_bilibili_errors -> fetching exception."""
    with pytest.raises(FetchingError) as ei:
        async with map_bilibili_errors(_LABEL):
            raise sdk_exc
    return ei.value


@pytest.mark.parametrize("sdk_exc", _SDK_CASES)
async def test_worker_pack_matches_direct_mapping(sdk_exc: BaseException) -> None:
    direct_exc = await _direct(sdk_exc)
    pack = map_sdk_exception(_LABEL, sdk_exc)

    # worker's type string and message match the directly-mapped exception exactly
    assert pack["type"] == type(direct_exc).__name__
    assert pack["message"] == str(direct_exc)


@pytest.mark.parametrize("sdk_exc", _SDK_CASES)
async def test_full_ipc_roundtrip_classification(sdk_exc: BaseException) -> None:
    """worker map -> JSON frame -> main rebuild -> classify == direct classify."""
    direct_exc = await _direct(sdk_exc)
    baseline = classify_fetching_exception(direct_exc)

    pack = map_sdk_exception(_LABEL, sdk_exc)
    # cross the wire as a real NDJSON frame
    frame = encode_frame({"id": 1, "status": "error", "error": pack})
    decoded = decode_frame(frame)["error"]
    rebuilt = fetching_exception_from_pack(ErrorPack.from_dict(decoded))

    assert type(rebuilt) is type(direct_exc)
    assert str(rebuilt) == str(direct_exc)
    assert classify_fetching_exception(rebuilt) == baseline


def test_response_code_carries_diagnostic_code() -> None:
    assert map_sdk_exception(_LABEL, ResponseCodeException(412, "x", {}))["code"] == 412
    assert map_sdk_exception(_LABEL, ResponseCodeException(53013, "x", {}))["code"] == 53013
    assert map_sdk_exception(_LABEL, NetworkException(404, "x"))["code"] == 404
    assert map_sdk_exception(_LABEL, TimeoutError())["code"] is None


def test_download_error_pack_is_permanent() -> None:
    """Audio download failure maps to permanent, not the retryable default (§6.4/§7.2)."""
    pack = download_error_pack("videos: no audio stream")
    assert pack["classification"] == "permanent"
    assert pack["retryable_hint"] is False
    # rebuilds on the main side as a permanent (ResourceUnavailable) fetching exception
    rebuilt = fetching_exception_from_pack(ErrorPack.from_dict(pack))
    assert classify_fetching_exception(rebuilt).name == "PERMANENT"


def test_protocol_error_pack_shape() -> None:
    pack = protocol_error_pack("unknown op: frobnicate")
    assert pack["type"] == "protocol_error"
    assert pack["classification"] == "permanent"
    # protocol_error is not a fetching type; ErrorPack.from_dict still validates the shape
    assert ErrorPack.from_dict(pack).type == "protocol_error"
