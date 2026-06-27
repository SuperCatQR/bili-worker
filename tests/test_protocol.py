"""Tests for the worker NDJSON protocol codec (contract §4)."""

from __future__ import annotations

import json

import pytest

from bili_worker.protocol import (
    ProtocolError,
    Request,
    Response,
    decode_frame,
    encode_frame,
    error_response,
    ok_response,
)


def test_frame_is_single_line_compact() -> None:
    obj = {"id": 1, "op": "fetch_page", "params": {"text": "line1\nline2", "n": 2}}
    frame = encode_frame(obj)
    assert frame.endswith("\n")
    assert frame.count("\n") == 1  # only the trailing terminator
    assert ": " not in frame  # compact separators, no spaces
    # the embedded payload newline is escaped, not a bare frame newline
    assert "\\n" in frame
    assert decode_frame(frame) == obj


def test_roundtrip_request() -> None:
    req = Request(id=42, op="fetch_page", params={"uid": 123, "endpoint": "videos"})
    frame = req.to_frame()
    parsed = Request.from_obj(decode_frame(frame))
    assert parsed == req


def test_request_defaults_empty_params() -> None:
    parsed = Request.from_obj({"id": 1, "op": "handshake"})
    assert parsed.params == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"op": "x", "params": {}},                 # missing id
        {"id": 1, "params": {}},                   # missing op
        {"id": "1", "op": "x"},                    # id not int
        {"id": True, "op": "x"},                   # bool is not a valid id
        {"id": 1, "op": 5},                        # op not str
        {"id": 1, "op": "x", "params": []},        # params not object
    ],
)
def test_request_malformed_raises(bad: dict) -> None:
    with pytest.raises(ProtocolError):
        Request.from_obj(bad)


def test_decode_frame_rejects_garbage() -> None:
    with pytest.raises(ProtocolError):
        decode_frame("not json")
    with pytest.raises(ProtocolError):
        decode_frame("")
    with pytest.raises(ProtocolError):
        decode_frame("[1,2,3]")  # array, not object


def test_ok_and_error_response_envelopes() -> None:
    ok = ok_response(7, {"raw_payload": {"a": 1}})
    assert ok == {"id": 7, "status": "ok", "data": {"raw_payload": {"a": 1}}}
    parsed_ok = Response.from_obj(json.loads(encode_frame(ok)))
    assert parsed_ok.status == "ok"
    assert parsed_ok.data == {"raw_payload": {"a": 1}}
    assert parsed_ok.error is None

    err_pack = {"type": "Http412Error", "classification": "retryable", "message": "ep: 412"}
    err = error_response(7, err_pack)
    parsed_err = Response.from_obj(json.loads(encode_frame(err)))
    assert parsed_err.status == "error"
    assert parsed_err.error == err_pack
    assert parsed_err.data is None


@pytest.mark.parametrize(
    "bad",
    [
        {"id": 1},                                  # missing status
        {"id": 1, "status": "weird"},               # invalid status
        {"id": 1, "status": "ok"},                  # ok without data
        {"id": 1, "status": "ok", "data": 5},       # data not object
        {"id": 1, "status": "error"},               # error without error obj
        {"id": 1, "status": "error", "error": "x"}, # error not object
    ],
)
def test_response_malformed_raises(bad: dict) -> None:
    with pytest.raises(ProtocolError):
        Response.from_obj(bad)
