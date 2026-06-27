"""In-memory FakeWorker for testing worker lifecycle scenarios (contract §9).

Provides a drop-in substitute for a real worker subprocess that uses asyncio
queues instead of real pipes.  Supports simulating crash scenarios (stdout EOF,
non-zero exit, heartbeat timeout) so tests can verify that in-flight requests
are classified correctly without spawning real processes.

This module is test infrastructure — it is NOT imported by production code.

ponytail: lazy-imports dispatch/errors inside _run_loop to avoid triggering
``bilibili_api`` import at module level (the SDK may not be installed in CI).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from bili_worker.protocol import (
    ProtocolError,
    Request,
    decode_frame,
    encode_frame,
    error_response,
)


class FakeWorker:
    """In-memory worker that processes requests through the real dispatch loop.

    Requests are fed via ``send_request(frame: str)`` and responses are read
    from ``read_response()``.  The worker runs an internal asyncio task that
    reads from an input queue, dispatches, and writes to an output queue.

    Crash simulation methods:
    - ``close_stdout()`` — simulates worker stdout EOF (process crash)
    - ``exit_nonzero()`` — simulates worker exiting with non-zero code
    - ``stop_heartbeat()`` — simulates heartbeat timeout (worker hung)
    """

    def __init__(self) -> None:
        self._in_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._out_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._exit_code: int = 0
        self._crashed: bool = False
        self._crash_reason: str = ""
        # Track in-flight request ids for crash-scenario verification.
        self._in_flight: set[int] = set()
        self._completed: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the internal dispatch loop task."""
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Graceful shutdown: send sentinel, await task."""
        if self._task is not None and not self._task.done():
            await self._in_queue.put(None)  # EOF sentinel
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

    # ------------------------------------------------------------------
    # Request / response
    # ------------------------------------------------------------------

    async def send_request(self, frame: str) -> None:
        """Feed a raw NDJSON frame (with trailing ``\\n``) to the worker."""
        await self._in_queue.put(frame)

    async def read_response(self, *, timeout: float = 5.0) -> str | None:
        """Read the next response frame from the worker.

        Returns ``None`` if the output stream is closed (stdout EOF / crash).
        """
        try:
            return await asyncio.wait_for(self._out_queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    # ------------------------------------------------------------------
    # Crash simulation (contract §9)
    # ------------------------------------------------------------------

    def close_stdout(self) -> None:
        """Simulate worker stdout EOF — process crashed mid-stream.

        Pushes ``None`` (EOF sentinel) into the output queue.  In-flight
        requests should be classified as ``Http5xxError`` / retryable.
        """
        self._crashed = True
        self._crash_reason = "stdout_eof"
        self._out_queue.put_nowait(None)

    def exit_nonzero(self, code: int = 1) -> None:
        """Simulate worker exiting with a non-zero exit code.

        Sets the exit code and closes stdout.  The main process should detect
        the non-zero exit and trigger restart logic (§9).
        """
        self._exit_code = code
        self._crashed = True
        self._crash_reason = f"exit_code_{code}"
        self._out_queue.put_nowait(None)

    def stop_heartbeat(self) -> None:
        """Simulate heartbeat timeout — worker is hung / unresponsive.

        The output queue is left open but no responses will be produced.
        The main process should detect the heartbeat timeout and classify
        in-flight requests as ``Http5xxError`` / retryable.
        """
        self._crashed = True
        self._crash_reason = "heartbeat_timeout"
        # Don't close stdout — just stop processing.  The internal loop
        # task is cancelled to simulate a hung worker.
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def exit_code(self) -> int:
        return self._exit_code

    @property
    def crashed(self) -> bool:
        return self._crashed

    @property
    def crash_reason(self) -> str:
        return self._crash_reason

    @property
    def in_flight_ids(self) -> frozenset[int]:
        return frozenset(self._in_flight)

    # ------------------------------------------------------------------
    # Internal dispatch loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Read frames from _in_queue, dispatch, write responses to _out_queue.

        Lazy-imports dispatch/errors to avoid triggering ``bilibili_api``
        import at module level (the SDK may not be installed in CI).
        """
        # Lazy imports — these modules import bilibili_api.
        try:
            from bili_worker.__main__ import dispatch  # noqa: PLC0415
            from bili_worker.errors import map_sdk_exception, protocol_error_pack  # noqa: PLC0415
        except ImportError as exc:
            # SDK not available — push an error frame and exit.
            err_frame = encode_frame(
                {"id": -1, "status": "error", "error": {
                    "type": "protocol_error",
                    "classification": "permanent",
                    "code": None,
                    "message": f"worker import error: {exc}",
                    "retryable_hint": False,
                }}
            )
            await self._out_queue.put(err_frame)
            return

        while True:
            line = await self._in_queue.get()
            if line is None:
                # stdin EOF — clean exit
                break

            try:
                obj = decode_frame(line)
                req = Request.from_obj(obj)
            except ProtocolError as exc:
                err_frame = encode_frame(
                    {"id": -1, "status": "error", "error": protocol_error_pack(str(exc))}
                )
                await self._out_queue.put(err_frame)
                continue

            self._in_flight.add(req.id)
            try:
                resp = await dispatch(req)
            except Exception:
                # dispatch() already catches ProtocolError and generic Exception
                # inside itself; this is a safety net.
                resp = error_response(req.id, map_sdk_exception(req.op, Exception("dispatch failed")))
            finally:
                self._in_flight.discard(req.id)

            self._completed[req.id] = resp
            await self._out_queue.put(encode_frame(resp))

            if req.op == "shutdown":
                break
