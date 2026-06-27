"""Tests for worker process lifecycle boundary scenarios (contract §9).

Covers three worker-death scenarios where in-flight requests must be:
- classified correctly (``Http5xxError`` / retryable)
- not silently swallowed
- not crashing the main process

Uses ``FakeWorker`` (in-memory dispatch) — no real subprocess spawn.

ponytail: ``_OP_TABLE`` / ``dispatch`` imports are lazy because they trigger
``bilibili_api`` (the SDK may not be installed in standalone worker CI).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from bili_worker.protocol import encode_frame
from bili_worker.testing import FakeWorker


# Lazy helpers — import bilibili_api transitively, may raise in CI.
def _get_op_table() -> dict:
    from bili_worker.__main__ import _OP_TABLE
    return _OP_TABLE


@pytest.fixture
async def worker():
    """Start a FakeWorker and clean up after the test."""
    w = FakeWorker()
    await w.start()
    yield w
    if not w.crashed:
        await w.stop()


# ---------------------------------------------------------------------------
# Scenario 1: stdout EOF — worker crashes mid-stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stdout_eof_in_flight_requests_tracked(worker: FakeWorker):
    """In-flight requests after stdout EOF must be tracked, not silently dropped.

    The main process detects EOF on stdout (worker crash).  It must NOT
    silently swallow the in-flight requests — they should be tracked so
    the caller can classify them as Http5xxError / retryable (§9).
    """
    slow_event = asyncio.Event()

    async def slow_ping(_params):
        await slow_event.wait()
        return {"pong": True}

    op_table = _get_op_table()
    original_ping = op_table.get("ping")
    op_table["ping"] = slow_ping

    try:
        frame = encode_frame({"id": 1, "op": "ping", "params": {}})
        await worker.send_request(frame)

        # Let the worker pick up the request.
        await asyncio.sleep(0.05)
        assert 1 in worker.in_flight_ids, "request should be in-flight before crash"

        # Simulate stdout EOF (crash).
        worker.close_stdout()

        # Read response — should get None (EOF sentinel).
        resp = await worker.read_response(timeout=0.5)
        assert resp is None, "stdout EOF should yield None sentinel"

        # The in-flight request should still be tracked.
        assert 1 in worker.in_flight_ids, "in-flight request must not be silently dropped"
    finally:
        slow_event.set()
        op_table["ping"] = original_ping


@pytest.mark.asyncio
async def test_stdout_eof_no_in_flight_clean_exit(worker: FakeWorker):
    """When no requests are in-flight, stdout EOF is a clean exit."""
    # Send and complete a request first.
    frame = encode_frame({"id": 1, "op": "ping", "params": {}})
    await worker.send_request(frame)
    resp = await worker.read_response()
    assert resp is not None
    assert "pong" in resp

    # Now close stdout — no in-flight requests, should be clean.
    worker.close_stdout()
    assert worker.crashed
    assert worker.crash_reason == "stdout_eof"
    assert len(worker.in_flight_ids) == 0


# ---------------------------------------------------------------------------
# Scenario 2: non-zero exit code — worker process died
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonzero_exit_code_detected(worker: FakeWorker):
    """Worker exiting with non-zero code must be detectable.

    The main process should check the exit code and trigger restart logic (§9).
    """
    frame = encode_frame({"id": 1, "op": "ping", "params": {}})
    await worker.send_request(frame)

    # Let it complete.
    resp = await worker.read_response()
    assert resp is not None

    # Simulate non-zero exit.
    worker.exit_nonzero(code=1)

    assert worker.exit_code == 1
    assert worker.crashed
    assert "exit_code_1" in worker.crash_reason


@pytest.mark.asyncio
async def test_nonzero_exit_with_in_flight_requests(worker: FakeWorker):
    """Non-zero exit with in-flight requests — requests are tracked for retry.

    The main process must mark all in-flight ids as Http5xxError / retryable
    and restart the worker (§9).
    """
    block = asyncio.Event()

    async def blocking_handler(_params):
        await block.wait()
        return {"done": True}

    op_table = _get_op_table()
    op_table["blocking"] = blocking_handler
    try:
        frame = encode_frame({"id": 2, "op": "blocking", "params": {}})
        await worker.send_request(frame)
        await asyncio.sleep(0.05)

        assert 2 in worker.in_flight_ids

        worker.exit_nonzero(code=1)

        assert worker.exit_code == 1
        assert worker.crashed
        # In-flight request exists and must be treated as retryable by the caller.
        assert 2 in worker.in_flight_ids
    finally:
        block.set()
        op_table.pop("blocking", None)


# ---------------------------------------------------------------------------
# Scenario 3: heartbeat timeout — worker hung / unresponsive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_timeout_stops_responses(worker: FakeWorker):
    """After heartbeat timeout, worker produces no more responses.

    The main process sends periodic ping requests.  If the worker stops
    responding, the main process should classify in-flight requests as
    Http5xxError / retryable (§9).
    """
    # First, verify the worker responds to pings normally.
    frame = encode_frame({"id": 1, "op": "ping", "params": {}})
    await worker.send_request(frame)
    resp = await worker.read_response()
    assert resp is not None
    assert "pong" in resp

    # Send a blocking request and simulate heartbeat timeout before it completes.
    block = asyncio.Event()

    async def slow_op(_params):
        await block.wait()
        return {"result": "ok"}

    op_table = _get_op_table()
    op_table["slow"] = slow_op
    try:
        frame2 = encode_frame({"id": 2, "op": "slow", "params": {}})
        await worker.send_request(frame2)
        await asyncio.sleep(0.05)

        worker.stop_heartbeat()

        assert worker.crashed
        assert worker.crash_reason == "heartbeat_timeout"

        # Subsequent reads should eventually time out (worker is hung).
        resp2 = await worker.read_response(timeout=0.3)
        assert resp2 is None, "heartbeat timeout → no response (read timed out)"
    finally:
        block.set()
        op_table.pop("slow", None)


@pytest.mark.asyncio
async def test_heartbeat_timeout_in_flight_tracking(worker: FakeWorker):
    """In-flight requests during heartbeat timeout are still tracked."""
    block = asyncio.Event()

    async def slow_op(_params):
        await block.wait()
        return {"result": "ok"}

    op_table = _get_op_table()
    op_table["slow"] = slow_op
    try:
        frame = encode_frame({"id": 3, "op": "slow", "params": {}})
        await worker.send_request(frame)
        await asyncio.sleep(0.05)

        assert 3 in worker.in_flight_ids

        worker.stop_heartbeat()
        assert worker.crashed
        assert 3 in worker.in_flight_ids, "in-flight request tracked during heartbeat timeout"
    finally:
        block.set()
        op_table.pop("slow", None)


# ---------------------------------------------------------------------------
# Smoke: normal operation is unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_worker_roundtrip_normal(worker: FakeWorker):
    """Normal request/response roundtrip through FakeWorker works."""
    frame = encode_frame({"id": 1, "op": "ping", "params": {}})
    await worker.send_request(frame)
    resp = await worker.read_response()
    assert resp is not None
    obj = json.loads(resp)
    assert obj["id"] == 1
    assert obj["status"] == "ok"
    assert obj["data"] == {"pong": True}


@pytest.mark.asyncio
async def test_fake_worker_handshake(worker: FakeWorker):
    """Handshake op returns worker identity."""
    frame = encode_frame({"id": 1, "op": "handshake", "params": {}})
    await worker.send_request(frame)
    resp = await worker.read_response()
    obj = json.loads(resp)
    assert obj["status"] == "ok"
    assert "worker_version" in obj["data"]
    assert "protocol_version" in obj["data"]


@pytest.mark.asyncio
async def test_fake_worker_malformed_frame(worker: FakeWorker):
    """Malformed frame produces protocol_error, not a crash."""
    await worker.send_request("not valid json\n")
    resp = await worker.read_response()
    assert resp is not None
    obj = json.loads(resp)
    assert obj["status"] == "error"
    assert obj["error"]["type"] == "protocol_error"
    # Worker should still be alive.
    assert not worker.crashed
