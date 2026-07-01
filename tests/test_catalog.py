"""Tests for the full op dispatch + 63-endpoint catalog (contract §5, §6, §11).

Covers the CHO-119 acceptance points that do NOT need a live bilibili network
call: catalog shape (count==63 / uid==33 / item==30), handshake fields, the
Windows-safe stdin loop, dispatch error routing (protocol_error vs AuthError
vs SDK-mapped), and the ``fetch_item`` ``parent_uid``-in-``extra`` line order
(CHO-101 review S2).
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator

import pytest

from bili_worker.__main__ import (
    dispatch,
    handle_describe_catalog,
    handle_fetch_item,
    handle_handshake,
    run_loop,
)
from bili_worker.credential import CredentialPool, credential_pool_clear
from bili_worker.protocol import Request


@pytest.fixture(autouse=True)
def _clean_credential_env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    credential_pool_clear()
    for key in (
        "BILI_SESSDATA", "BILI_JCT", "BILI_BUVID3", "BILI_BUVID4",
        "BILI_DEDEUSERID", "BILI_AC_TIME_VALUE", "BILI_ENV_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    credential_pool_clear()


# ---------------------------------------------------------------------------
# describe_catalog — contract §6.2 (main side validates count/uid/item)
# ---------------------------------------------------------------------------


async def test_describe_catalog_shape() -> None:
    data = await handle_describe_catalog({})
    assert data["count"] == 63
    assert data["uid_level"] == 33
    assert data["item_level"] == 30
    assert len(data["endpoints"]) == 63
    # serializable fields only — no callable / extract_items / skip_item leak
    keys = set(data["endpoints"][0])
    assert "callable" not in keys
    assert "extract_items" not in keys
    assert "skip_item" not in keys
    # needs_parent_uid endpoints are exactly the 3 known ones
    needs_uid = sorted(e["name"] for e in data["endpoints"] if e["needs_parent_uid"])
    assert needs_uid == ["channel_videos_season", "channel_videos_series", "upower_qa_detail"]


# ---------------------------------------------------------------------------
# handshake — contract §6.1 (worker_version / bilibili_api_version / capabilities)
# ---------------------------------------------------------------------------


async def test_handshake_fields() -> None:
    data = await handle_handshake({})
    assert data["protocol_version"] == "1.0"
    assert data["worker_version"]
    assert data["bilibili_api_version"] != "unknown"  # SDK is installed in worker env
    assert {"fetch_page", "fetch_item", "resolve_audio_url", "credential_ref"} <= set(data["capabilities"])


# ---------------------------------------------------------------------------
# dispatch error routing — contract §4.2 / §6.5 / §7
# ---------------------------------------------------------------------------


async def test_dispatch_unknown_op_is_protocol_error() -> None:
    resp = await dispatch(Request(id=1, op="frobnicate", params={}), CredentialPool())
    assert resp == {
        "id": 1,
        "status": "error",
        "error": {"type": "protocol_error", "classification": "permanent", "code": None,
                  "message": "unknown op: 'frobnicate'", "retryable_hint": False},
    }


async def test_dispatch_fetch_item_unknown_endpoint_is_protocol_error() -> None:
    resp = await dispatch(
        Request(id=2, op="fetch_item", params={"endpoint": "no_such", "item_id": "BV1"}),
        CredentialPool(),
    )
    assert resp["status"] == "error"
    assert resp["error"]["type"] == "protocol_error"


async def test_dispatch_kind_mismatch_is_protocol_error() -> None:
    # fetch_page on an item-kind endpoint
    r1 = await dispatch(
        Request(id=3, op="fetch_page", params={"uid": 1, "endpoint": "video_detail", "request_params": {}}),
        CredentialPool(),
    )
    assert r1["error"]["type"] == "protocol_error"
    # fetch_item on a uid-kind endpoint
    r2 = await dispatch(
        Request(id=4, op="fetch_item", params={"endpoint": "videos", "item_id": "BV1"}),
        CredentialPool(),
    )
    assert r2["error"]["type"] == "protocol_error"


async def test_dispatch_credential_open_missing_sessdata_is_auth_error() -> None:
    """No .env / no BILI_SESSDATA → AuthError/permanent (contract §6.5, §8)."""
    resp = await dispatch(
        Request(id=5, op="credential_open", params={"env_path": "/nonexistent/.env"}),
        CredentialPool(),
    )
    assert resp["status"] == "error"
    assert resp["error"]["type"] == "AuthError"
    assert resp["error"]["classification"] == "permanent"
    # message is redacted (no plaintext)
    assert "sessdata" not in str(resp["error"]["message"]).lower()


async def test_dispatch_credential_open_with_env_returns_ref(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("BILI_SESSDATA=s\nBILI_JCT=j\n", encoding="utf-8")
    resp = await dispatch(
        Request(id=6, op="credential_open", params={"env_path": str(env)}),
        CredentialPool(),
    )
    assert resp["status"] == "ok"
    assert resp["data"]["credential_ref"].startswith("cred-")
    assert resp["data"]["has_sessdata"] is True


# ---------------------------------------------------------------------------
# fetch_item parent_uid line order — CHO-101 review S2
# ---------------------------------------------------------------------------


async def test_fetch_item_reads_parent_uid_from_extra() -> None:
    """The main side nests parent_uid inside params.extra (CHO-101); the worker
    must unpack it from extra, not a top-level field.

    Network-free proof via ``_call_item`` directly: ``upower_qa_detail``
    (needs_parent_uid) does ``uid = int(kw["_uid"])``. With ``parent_uid=None``
    the ``_uid`` key is absent → ``KeyError`` → ``ValueError`` mentioning _uid.
    That confirms the only path by which parent_uid reaches the callable is the
    ``extra`` → ``_uid`` handoff wired in ``handle_fetch_item`` / ``_call_item``.
    """
    from bili_worker._catalog import _call_item

    with pytest.raises(ValueError, match="_uid"):
        await _call_item("upower_qa_detail", "1", None, None, 30.0)
    # non-needs_parent_uid endpoints ignore parent_uid entirely (no _uid read)
    with pytest.raises(ValueError, match="article_detail"):
        await _call_item("article_detail", "not-an-int", None, 42, 30.0)


async def test_fetch_item_ignores_top_level_parent_uid() -> None:
    """A top-level parent_uid (wrong line order) must NOT be consumed — the
    worker only reads extra.parent_uid."""
    pool = CredentialPool()
    # top-level parent_uid, no extra → _uid is None → ValueError mentioning _uid
    with pytest.raises(ValueError, match="_uid"):
        await handle_fetch_item(
            {"endpoint": "upower_qa_detail", "item_id": "1", "parent_uid": 42},
            pool,
        )


# ---------------------------------------------------------------------------
# Windows-safe stdin loop — contract §4 (the WinError 6 fix, CHO-119)
# ---------------------------------------------------------------------------


async def test_run_loop_handles_handshake_over_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_loop reads NDJSON frames via run_in_executor (not connect_read_pipe),
    so it works under the Windows Proactor event loop. Feed a handshake + EOF."""
    frame = '{"id":1,"op":"handshake","params":{}}\n'

    class _FakeStdin:
        def __init__(self, data: str) -> None:
            self._data = data

        def readline(self) -> str:
            if not self._data:
                return ""  # EOF
            line, _, self._data = self._data.partition("\n")
            return line + "\n"

    captured: list[str] = []

    class _FakeStdout:
        def write(self, s: str) -> None:
            captured.append(s)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdin", _FakeStdin(frame))
    monkeypatch.setattr(sys, "stdout", _FakeStdout())

    await run_loop()

    assert captured, "worker wrote no response"
    resp = json.loads(captured[0])
    assert resp["id"] == 1
    assert resp["status"] == "ok"
    assert resp["data"]["protocol_version"] == "1.0"


async def test_run_loop_malformed_frame_keeps_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad frame yields an error response but the loop keeps reading (§4.2)."""
    data = "not json\n" + '{"id":7,"op":"ping","params":{}}\n'

    class _FakeStdin:
        def __init__(self, d: str) -> None:
            self._data = d

        def readline(self) -> str:
            if not self._data:
                return ""
            line, _, self._data = self._data.partition("\n")
            return line + "\n"

    captured: list[str] = []

    class _FakeStdout:
        def write(self, s: str) -> None:
            captured.append(s)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdin", _FakeStdin(data))
    monkeypatch.setattr(sys, "stdout", _FakeStdout())

    await run_loop()

    assert len(captured) == 2
    err = json.loads(captured[0])
    assert err["status"] == "error" and err["error"]["type"] == "protocol_error"
    ok = json.loads(captured[1])
    assert ok["id"] == 7 and ok["status"] == "ok" and ok["data"]["pong"] is True
