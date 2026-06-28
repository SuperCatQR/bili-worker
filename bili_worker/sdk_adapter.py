"""Worker-side SDK adapter — port of bili_unit.fetching._bilibili_adapter + _adapter_core + _adapters.

This module imports ``bilibili_api`` and lives **only** in the worker process.
The main process never imports this code — it talks to the worker via stdio JSON.

Contains:
- Error mapping context manager (internal use) + error-pack builder
- json_safe / normalise_api_result (for serialising API returns)
- Pagination helpers & strategies
- 63 endpoint callable definitions (uid-level + item-level fan-out)
- Video / subtitle / article / opus / channel adapters
- Audio URL resolver
- Login helpers
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

import aiohttp
from bilibili_api import Credential, request_settings, select_client, user
from bilibili_api.article import Article, ArticleList
from bilibili_api.channel_series import ChannelOrder
from bilibili_api.exceptions import (
    ApiException,
    ArgsException,
    CredentialNoBiliJctException,
    CredentialNoSessdataException,
    InitialStateException,
    NetworkException,
    ResponseCodeException,
)
from bilibili_api.opus import Opus
from bilibili_api.video import AudioQuality, Video, VideoDownloadURLDataDetecter

# ---------------------------------------------------------------------------
# Internal exception types (mirror bili_unit.fetching hierarchy)
# ---------------------------------------------------------------------------


class _FetchingError(Exception):
    """Base for all internal fetching exceptions (not serialised to main)."""


class _Http5xxError(_FetchingError):
    """Server-side error."""


class _Http412Error(_FetchingError):
    """412 — too many requests."""


class _AuthError(_FetchingError):
    """Auth failure."""


class _InvalidRequestError(_FetchingError):
    """Invalid SDK args."""


class _ResourceUnavailableError(_FetchingError):
    """Permanent business failure."""


class _RequestError(_FetchingError):
    """Generic request-level failure."""


# ---------------------------------------------------------------------------
# Error-pack builder (for dispatch to serialise internal errors to wire format)
# ---------------------------------------------------------------------------

def _pack_error(type_: str, classification: str, message: str, code: int | None) -> dict[str, Any]:
    return {
        "type": type_,
        "classification": classification,
        "code": code,
        "message": message,
        "retryable_hint": classification == "retryable",
    }


def map_internal_exception(op: str, exc: BaseException) -> dict[str, Any]:
    """Map an internal fetching exception (from _map_bilibili_errors) to an error pack."""
    if isinstance(exc, _Http5xxError):
        return _pack_error("Http5xxError", "retryable", f"{op}: {exc}", None)
    if isinstance(exc, _Http412Error):
        return _pack_error("Http412Error", "retryable", f"{op}: {exc}", 412)
    if isinstance(exc, _AuthError):
        return _pack_error("AuthError", "permanent", f"{op}: {exc}", None)
    if isinstance(exc, _InvalidRequestError):
        return _pack_error("InvalidRequestError", "permanent", f"{op}: {exc}", None)
    if isinstance(exc, _ResourceUnavailableError):
        return _pack_error("ResourceUnavailableError", "unavailable", f"{op}: {exc}", None)
    if isinstance(exc, _RequestError):
        return _pack_error("RequestError", "retryable", f"{op}: {exc}", None)
    if isinstance(exc, TimeoutError):
        return _pack_error("Http5xxError", "retryable", f"{op}: timeout: {exc}", None)
    return _pack_error("RequestError", "retryable", f"{op}: unexpected: {exc}", None)


# ---------------------------------------------------------------------------
# Error mapping context manager
# ---------------------------------------------------------------------------

_PERMANENT_BUSINESS_CODES: frozenset[int] = frozenset({
    -400, 22115, 22118, 53013, 53016, 88214,
})


@contextlib.asynccontextmanager
async def _map_bilibili_errors(label: str, *, passthrough: tuple[type[BaseException], ...] = ()):
    """Map bilibili-api exceptions onto internal fetching exceptions."""
    try:
        yield
    except TimeoutError as exc:
        raise _Http5xxError(f"{label}: timeout") from exc
    except ResponseCodeException as exc:
        if exc.code == 412:
            raise _Http412Error(f"{label}: 412") from exc
        if exc.code in _PERMANENT_BUSINESS_CODES:
            raise _ResourceUnavailableError(f"{label}: code={exc.code}: {exc.msg}") from exc
        raise _RequestError(f"{label}: code={exc.code}: {exc.msg}") from exc
    except NetworkException as exc:
        status = getattr(exc, "status", 0) or 0
        if status == 404:
            raise _ResourceUnavailableError(f"{label}: HTTP 404 (route gone): {exc}") from exc
        if 400 <= status < 500:
            raise _RequestError(f"{label}: HTTP {status}: {exc}") from exc
        raise _Http5xxError(f"{label}: network error {exc}") from exc
    except passthrough:
        raise
    except (CredentialNoSessdataException, CredentialNoBiliJctException) as exc:
        raise _AuthError(f"{label}: credential missing: {exc}") from exc
    except ArgsException as exc:
        raise _InvalidRequestError(f"{label}: invalid SDK arguments: {exc}") from exc
    except ApiException as exc:
        raise _RequestError(f"{label}: {exc}") from exc
    except Exception as exc:
        raise _RequestError(f"{label}: unexpected: {exc}") from exc


# ---------------------------------------------------------------------------
# json_safe — convert bilibili-api return objects into JSON-serialisable values
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): json_safe(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return str(value)


def normalise_api_result(result: Any, key: str = "data") -> dict[str, Any]:
    safe = json_safe(result)
    if isinstance(safe, dict):
        return safe
    return {key: safe}


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------

def resolve_dot_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for seg in path.split("."):
        if not seg:
            continue
        if isinstance(current, dict) and seg in current:
            current = current[seg]
        else:
            return None
    return current


def extract_list_items(data: dict[str, Any], path: str | None = None) -> list:
    container: Any = resolve_dot_path(data, path) if path else None
    if isinstance(container, list):
        return container
    if isinstance(container, dict):
        for k in ("list", "items", "archives", "data", "media_list"):
            value = container.get(k)
            if isinstance(value, list):
                return value
        collected: list = []
        for value in container.values():
            if isinstance(value, list):
                collected.extend(value)
        return collected
    for k in ("list", "items", "archives", "data", "media_list", "cards"):
        value = data.get(k)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for nested in ("list", "items", "archives", "data", "media_list"):
                nested_value = value.get(nested)
                if isinstance(nested_value, list):
                    return nested_value
    return []


def extract_total_count(data: dict[str, Any]) -> int:
    for path in ("page.count", "page.total", "items_lists.page.total", "total", "count", "total_count", "totalSize"):
        value = resolve_dot_path(data, path)
        if isinstance(value, int):
            return value
    return 0


# ---------------------------------------------------------------------------
# Pagination strategies
# ---------------------------------------------------------------------------

def _paginate_none(spec: Any, data: dict[str, Any], request_params: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    return True, None


def _paginate_page(spec: Any, data: dict[str, Any], request_params: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    total_count = extract_total_count(data)
    pi = data.get("page")
    if isinstance(pi, dict):
        total_count = total_count or pi.get("count", 0)
    if total_count == 0 and "totalSize" in data:
        total_count = data.get("totalSize", 0)
    if total_count == 0:
        il_page = resolve_dot_path(data, "items_lists.page")
        if isinstance(il_page, dict):
            total_count = il_page.get("total", 0)
    if total_count == 0 and "count" in data and isinstance(data["count"], int):
        total_count = data["count"]
    if total_count == 0 and "total_count" in data and isinstance(data["total_count"], int):
        total_count = data["total_count"]
    items = extract_list_items(data, spec.items_path)
    current_pn = request_params.get("pn", 1)
    ps = request_params.get("ps", 30)
    if not items or (total_count > 0 and current_pn * ps >= total_count):
        return True, None
    return False, {**request_params, "pn": current_pn + 1}


def _paginate_cursor(spec: Any, data: dict[str, Any], request_params: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    has_more = data.get("has_more", 0) == 1
    if not has_more:
        return True, None
    return False, {**request_params, "offset": data.get("offset", "")}


def _paginate_anchor(spec: Any, data: dict[str, Any], request_params: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    anchor = data.get("anchor", 0)
    if not anchor:
        return True, None
    return False, {**request_params, "anchor": anchor}


def _paginate_legacy_offset(spec: Any, data: dict[str, Any], request_params: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    next_offset = data.get("next_offset", 0)
    has_more = data.get("has_more", 0) == 1
    if not has_more or not next_offset:
        return True, None
    return False, {**request_params, "offset": next_offset}


def _paginate_oid(spec: Any, data: dict[str, Any], request_params: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    items = extract_list_items(data, spec.items_path)
    ps = request_params.get("ps", 100)
    total_count = extract_total_count(data)
    if not items or (total_count > 0 and len(items) >= total_count):
        return True, None
    last = items[-1] if isinstance(items[-1], dict) else {}
    next_oid = last.get("aid") or last.get("id") or last.get("oid") or last.get("param")
    if not next_oid or len(items) < ps:
        return True, None
    return False, {**request_params, "oid": next_oid}


_PAGINATION_STRATEGIES: dict[str, Callable] = {
    "none": _paginate_none,
    "page": _paginate_page,
    "cursor": _paginate_cursor,
    "anchor": _paginate_anchor,
    "legacy_offset": _paginate_legacy_offset,
    "oid": _paginate_oid,
}


# ---------------------------------------------------------------------------
# fetch_endpoint — called by __main__.py for fetch_page op
# ---------------------------------------------------------------------------

async def fetch_endpoint(
    uid: int,
    spec: Any,  # EndpointSpec from catalog
    credential: Credential | None,
    request_params: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call one page of an endpoint and return raw_payload + pagination info."""
    async with _map_bilibili_errors(spec.name):
        data = await asyncio.wait_for(
            spec.callable(uid, cred=credential, **request_params),
            timeout=timeout,
        )
    safe_data = json_safe(data)
    if not isinstance(safe_data, dict):
        safe_data = {"data": safe_data}
    strategy_fn = _PAGINATION_STRATEGIES[spec.pagination_strategy]
    is_last, next_req = strategy_fn(spec, safe_data, request_params)
    return {
        "raw_payload": safe_data,
        "is_last_page": is_last,
        "next_request": next_req,
    }


# ---------------------------------------------------------------------------
# fetch_item
# ---------------------------------------------------------------------------

async def fetch_item(
    item_id: str,
    spec: Any,
    credential: Credential | None,
    extra: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call one item-level endpoint and return raw_payload."""
    kwargs: dict[str, Any] = dict(extra or {})
    async with _map_bilibili_errors(spec.name):
        data = await asyncio.wait_for(
            spec.callable(item_id, credential=credential, **kwargs),
            timeout=timeout,
        )
    return {"raw_payload": json_safe(data)}


# ---------------------------------------------------------------------------
# resolve_audio_url
# ---------------------------------------------------------------------------

def _resolve_quality(quality: str):
    mapping = {
        "64K": AudioQuality._64K,
        "132K": AudioQuality._132K,
        "192K": getattr(AudioQuality, "_192K", AudioQuality._132K),
        "dolby": getattr(AudioQuality, "DOLBY", AudioQuality._132K),
        "hires": getattr(AudioQuality, "HI_RES", AudioQuality._132K),
    }
    return mapping.get(quality.upper().replace(" ", ""), AudioQuality._64K)


def _extract_duration(data: dict) -> float | None:
    for k in ("dash", "data"):
        inner = data.get(k)
        if isinstance(inner, dict):
            dur = inner.get("duration")
            if dur is not None:
                return float(dur) / 1000.0
    dur = data.get("duration")
    if dur is not None:
        return float(dur)
    return None


async def resolve_audio_url(
    bvid: str,
    credential: Credential | None,
    audio_quality: int | None = None,
) -> dict[str, Any]:
    """Resolve the CDN audio stream URL for a video."""
    video = Video(bvid=bvid, credential=credential)
    async with _map_bilibili_errors(f"resolve_audio_url[{bvid}]"):
        data = await video.get_download_url(page_index=0)
        detecter = VideoDownloadURLDataDetecter(data)
        quality_str = "64K"
        if audio_quality is not None:
            quality_map = {0: "64K", 1: "132K", 2: "192K", 3: "dolby", 4: "hires"}
            quality_str = quality_map.get(audio_quality, "64K")
        quality_enum = _resolve_quality(quality_str)
        streams = detecter.detect(audio_max_quality=quality_enum)
        audio_stream = None
        for s in streams:
            if type(s).__name__ == "AudioStreamDownloadURL":
                audio_stream = s
                break
        if audio_stream is None:
            raise _ResourceUnavailableError(f"no audio stream found for {bvid}")
        duration = _extract_duration(data)
    return {
        "url": audio_stream.url,
        "quality": str(getattr(audio_stream, "audio_quality", quality_str)),
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# HTTP backend bootstrap
# ---------------------------------------------------------------------------

def init_http_backend(backend: str = "aiohttp", impersonate: str = "chrome131") -> None:
    try:
        import curl_cffi  # noqa: F401
        if backend == "curl_cffi":
            select_client("curl_cffi")
            request_settings.set("impersonate", impersonate)
            return
    except ImportError:
        pass
    select_client("aiohttp")


# ---------------------------------------------------------------------------
# _user_method helper
# ---------------------------------------------------------------------------

def _user_method(name: str, **defaults: Any):
    def _fn(uid, cred=None, **kw):
        merged = {**defaults, **kw}
        return getattr(user.User(uid, credential=cred), name)(**merged)
    return _fn


# ---------------------------------------------------------------------------
# Scalar / list wrapping helpers
# ---------------------------------------------------------------------------

async def _wrap_scalar_result(coro: Awaitable, key: str = "value") -> dict:
    result = await coro
    safe = json_safe(result)
    if isinstance(safe, dict):
        return safe
    return {key: safe}


async def _wrap_list_result(coro: Awaitable) -> dict:
    return normalise_api_result(await coro, key="list")


# ---------------------------------------------------------------------------
# _extract_bvids_from_videos
# ---------------------------------------------------------------------------

def _extract_bvids_from_videos(raw_payload: dict) -> list[str]:
    bvids: list[str] = []
    for page in raw_payload.get("pages", []):
        vlist = page.get("list", {}).get("vlist", [])
        for item in vlist:
            bvid = item.get("bvid")
            if bvid:
                bvids.append(bvid)
    return bvids


# ---------------------------------------------------------------------------
# Video adapters
# ---------------------------------------------------------------------------

async def fetch_video_detail_item(bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
    v = Video(bvid, credential=credential)
    async with _map_bilibili_errors(f"video_detail[{bvid}]: get_info"):
        info = await asyncio.wait_for(v.get_info(), timeout=timeout)
    async with _map_bilibili_errors(f"video_detail[{bvid}]: get_tags"):
        tags = await asyncio.wait_for(v.get_tags(), timeout=timeout)
    return {"info": info, "tags": tags}


async def _video_pages(v: Video, bvid: str, timeout: float) -> list[dict[str, Any]]:
    async with _map_bilibili_errors(f"video[{bvid}]: get_pages"):
        pages = await asyncio.wait_for(v.get_pages(), timeout=timeout)
    return json_safe(pages)


def _video_item_method(method_name: str, *, per_page: bool = False, page_arg: str = "cid", result_key: str | None = None, default_kwargs: dict[str, Any] | None = None):
    async def _fn(bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
        v = Video(bvid, credential=credential)
        key = result_key or method_name
        kwargs = dict(default_kwargs or {})
        if not per_page:
            async with _map_bilibili_errors(f"{key}[{bvid}]"):
                result = await asyncio.wait_for(getattr(v, method_name)(**kwargs), timeout=timeout)
            return {key: json_safe(result)}
        pages = await _video_pages(v, bvid, timeout)
        rows: list[dict[str, Any]] = []
        for idx, page in enumerate(pages):
            call_kwargs = dict(kwargs)
            cid = page.get("cid") if isinstance(page, dict) else None
            if page_arg == "cid":
                call_kwargs["cid"] = cid
            elif page_arg == "page_index":
                call_kwargs["page_index"] = idx
            elif page_arg == "both":
                call_kwargs["cid"] = cid
                call_kwargs["page_index"] = idx
            async with _map_bilibili_errors(f"{key}[{bvid}][{idx}]"):
                result = await asyncio.wait_for(getattr(v, method_name)(**call_kwargs), timeout=timeout)
            rows.append({"page_index": idx, "cid": cid, "part": page.get("part", "") if isinstance(page, dict) else "", "result": json_safe(result)})
        return {"pages": pages, key: rows}
    return _fn


fetch_video_pages_item = _video_item_method("get_pages", result_key="pages")
fetch_video_detail_full_item = _video_item_method("get_detail", result_key="detail")
fetch_video_ai_conclusion_item = _video_item_method("get_ai_conclusion", per_page=True, page_arg="both", result_key="ai_conclusion")
fetch_video_danmaku_snapshot_item = _video_item_method("get_danmaku_snapshot", result_key="danmaku_snapshot")
fetch_video_danmaku_view_item = _video_item_method("get_danmaku_view", per_page=True, page_arg="both", result_key="danmaku_view")
fetch_video_danmaku_xml_item = _video_item_method("get_danmaku_xml", per_page=True, page_arg="both", result_key="danmaku_xml")
fetch_video_danmakus_item = _video_item_method("get_danmakus", per_page=True, page_arg="both", result_key="danmakus")
fetch_video_online_item = _video_item_method("get_online", per_page=True, page_arg="both", result_key="online")
fetch_video_pay_coins_item = _video_item_method("get_pay_coins", result_key="pay_coins")
fetch_video_pbp_item = _video_item_method("get_pbp", per_page=True, page_arg="both", result_key="pbp")
fetch_video_player_info_item = _video_item_method("get_player_info", per_page=True, page_arg="cid", result_key="player_info")
fetch_video_private_notes_item = _video_item_method("get_private_notes_list", result_key="private_notes")
fetch_video_related_item = _video_item_method("get_related", result_key="related")
fetch_video_relation_item = _video_item_method("get_relation", result_key="relation")
fetch_video_special_dms_item = _video_item_method("get_special_dms", per_page=True, page_arg="both", result_key="special_dms")
fetch_video_up_mid_item = _video_item_method("get_up_mid", result_key="up_mid")
fetch_video_snapshot_item = _video_item_method("get_video_snapshot", per_page=True, page_arg="cid", result_key="video_snapshot", default_kwargs={"json_index": True, "pvideo": False})
fetch_video_download_url_item = _video_item_method("get_download_url", per_page=True, page_arg="both", result_key="download_url")
fetch_video_is_episode_item = _video_item_method("is_episode", result_key="is_episode")
fetch_video_is_forbid_note_item = _video_item_method("is_forbid_note", result_key="is_forbid_note")
fetch_video_chargers_item = _video_item_method("get_chargers", result_key="chargers")


async def fetch_video_public_notes_item(bvid: str, credential: Credential | None, timeout: float = 30.0, ps: int = 50, **_kw: Any) -> dict[str, Any]:
    v = Video(bvid, credential=credential)
    pages: list[dict[str, Any]] = []
    pn = 1
    while True:
        async with _map_bilibili_errors(f"public_notes[{bvid}][{pn}]"):
            data = await asyncio.wait_for(v.get_public_notes_list(pn=pn, ps=ps), timeout=timeout)
        safe = normalise_api_result(data)
        pages.append(safe)
        items = extract_list_items(safe)
        total = extract_total_count(safe)
        if not items or (total > 0 and pn * ps >= total) or (total == 0 and len(items) < ps):
            break
        pn += 1
    return {"pages": pages}


# ---------------------------------------------------------------------------
# Subtitle adapter
# ---------------------------------------------------------------------------

_SUBTITLE_FETCH_TIMEOUT = 10.0


def _normalise_subtitle_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "https:" + url
    return url


async def _fetch_subtitle_body(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    try:
        normalised = _normalise_subtitle_url(url)
        async with session.get(normalised) as resp:
            if resp.status != 200:
                return {"_fetch_error": f"http {resp.status}"}
            data = await resp.json(content_type=None)
    except TimeoutError:
        return {"_fetch_error": "timeout"}
    except aiohttp.ClientError as exc:
        return {"_fetch_error": f"client error: {exc}"}
    except (ValueError, TypeError) as exc:
        return {"_fetch_error": f"json parse: {exc}"}
    if not isinstance(data, dict):
        return {"_fetch_error": "unexpected shape: not a dict"}
    body = data.get("body")
    if not isinstance(body, list):
        return {"_fetch_error": "unexpected shape: missing body"}
    return {"body": body}


async def _missing_url_error() -> dict[str, Any]:
    return {"_fetch_error": "missing subtitle_url"}


async def fetch_video_subtitle_item(bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
    v = Video(bvid, credential=credential)
    pages = await _video_pages(v, bvid, timeout)
    rows: list[dict[str, Any]] = []
    client_timeout = aiohttp.ClientTimeout(total=_SUBTITLE_FETCH_TIMEOUT)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        for idx, page in enumerate(pages):
            cid = page.get("cid") if isinstance(page, dict) else None
            part = page.get("part", "") if isinstance(page, dict) else ""
            async with _map_bilibili_errors(f"video_subtitle[{bvid}][{idx}]"):
                index = await asyncio.wait_for(v.get_subtitle(cid=cid), timeout=timeout)
            index_safe = json_safe(index)
            subtitles = []
            if isinstance(index_safe, dict):
                raw_subs = index_safe.get("subtitles")
                if isinstance(raw_subs, list):
                    subtitles = raw_subs
            if subtitles:
                tasks = [
                    _fetch_subtitle_body(session, sub.get("subtitle_url", ""))
                    if isinstance(sub, dict) and sub.get("subtitle_url")
                    else _missing_url_error()
                    for sub in subtitles
                ]
                fetched = await asyncio.gather(*tasks)
                content: list[dict[str, Any]] = []
                for sub, fetch_result in zip(subtitles, fetched, strict=False):
                    entry: dict[str, Any] = {
                        "lan": sub.get("lan") if isinstance(sub, dict) else None,
                        "lan_doc": sub.get("lan_doc") if isinstance(sub, dict) else None,
                    }
                    entry.update(fetch_result)
                    content.append(entry)
            else:
                content = []
            rows.append({"page_index": idx, "cid": cid, "part": part, "result": index_safe, "content": content})
    return {"pages": pages, "subtitle": rows}


# ---------------------------------------------------------------------------
# Article adapters
# ---------------------------------------------------------------------------

def _extract_cvids_from_articles(raw_payload: dict) -> list[str]:
    cvids: list[str] = []
    for page in raw_payload.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        for art in page.get("articles", []) or []:
            if not isinstance(art, dict):
                continue
            cvid = art.get("id")
            if cvid is not None:
                cvids.append(str(cvid))
    return cvids


async def fetch_article_detail_item(cvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
    try:
        cvid_int = int(cvid)
    except (TypeError, ValueError) as exc:
        raise _InvalidRequestError(f"article_detail[{cvid}]: invalid cvid: {exc}") from exc
    a = Article(cvid_int, credential=credential)
    async with _map_bilibili_errors(f"article_detail[{cvid}]: get_info"):
        info = await asyncio.wait_for(a.get_info(), timeout=timeout)
    try:
        async with _map_bilibili_errors(f"article_detail[{cvid}]: fetch_content", passthrough=(InitialStateException, KeyError)):
            await asyncio.wait_for(a.fetch_content(), timeout=timeout)
            markdown_text: str = a.markdown()
            content_json: list[Any] = a.json()
    except InitialStateException as exc:
        raise _ResourceUnavailableError(f"article_detail[{cvid}]: fetch_content {exc} (article unavailable)") from exc
    except KeyError as exc:
        raise _ResourceUnavailableError(f"article_detail[{cvid}]: fetch_content missing key {exc} (article unavailable)") from exc
    return {"info": info, "markdown": markdown_text, "content_json": content_json}


def _extract_rlids_from_article_list(raw_payload: dict) -> list[str]:
    ids: list[str] = []
    for lst in raw_payload.get("lists", []) or []:
        if not isinstance(lst, dict):
            continue
        rlid = lst.get("id")
        if rlid is not None:
            ids.append(str(rlid))
    return ids


async def fetch_article_list_detail_item(rlid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
    try:
        rlid_int = int(rlid)
    except (TypeError, ValueError) as exc:
        raise _InvalidRequestError(f"article_list_detail[{rlid}]: invalid rlid: {exc}") from exc
    al = ArticleList(rlid_int, credential=credential)
    async with _map_bilibili_errors(f"article_list_detail[{rlid}]: get_content"):
        result = await asyncio.wait_for(al.get_content(), timeout=timeout)
    return result


# ---------------------------------------------------------------------------
# Opus adapters
# ---------------------------------------------------------------------------

def _extract_opus_ids_from_opus(raw_payload: dict) -> list[str]:
    ids: list[str] = []
    for page in raw_payload.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        for it in page.get("items", []) or []:
            if not isinstance(it, dict):
                continue
            oid = it.get("opus_id")
            if oid is not None:
                ids.append(str(oid))
    return ids


async def fetch_opus_detail_item(opus_id: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
    try:
        opus_id_int = int(opus_id)
    except (TypeError, ValueError) as exc:
        raise _InvalidRequestError(f"opus_detail[{opus_id}]: invalid opus_id: {exc}") from exc
    o = Opus(opus_id_int, credential=credential)
    try:
        async with _map_bilibili_errors(f"opus_detail[{opus_id}]: get_info", passthrough=(ApiException,)):
            info = await asyncio.wait_for(o.get_info(), timeout=timeout)
    except ApiException as exc:
        if "opus_id 不正确" in str(exc) or "fallback" in str(exc).lower():
            raise _ResourceUnavailableError(f"opus_detail[{opus_id}]: opus unavailable ({exc})") from exc
        raise _RequestError(f"opus_detail[{opus_id}]: get_info {exc}") from exc
    try:
        async with _map_bilibili_errors(f"opus_detail[{opus_id}]: markdown", passthrough=(KeyError,)):
            markdown_text: str = await asyncio.wait_for(o.markdown(), timeout=timeout)
            images: list[dict[str, Any]] = await asyncio.wait_for(o.get_images_raw_info(), timeout=timeout)
    except KeyError as exc:
        raise _ResourceUnavailableError(f"opus_detail[{opus_id}]: markdown missing key {exc} (opus unavailable)") from exc
    return {"info": info, "markdown": markdown_text, "images": images}


# ---------------------------------------------------------------------------
# Channel adapters
# ---------------------------------------------------------------------------

def _extract_season_ids(raw_payload: dict) -> list[str]:
    ids: list[str] = []
    for page in raw_payload.get("pages", []):
        items_lists = page.get("items_lists", {})
        for item in items_lists.get("seasons_list", []):
            meta = item.get("meta", {})
            sid = meta.get("season_id")
            if sid is not None:
                ids.append(str(sid))
    return ids


def _extract_series_ids(raw_payload: dict) -> list[str]:
    ids: list[str] = []
    for page in raw_payload.get("pages", []):
        items_lists = page.get("items_lists", {})
        for item in items_lists.get("series_list", []):
            meta = item.get("meta", {})
            sid = meta.get("series_id")
            if sid is not None:
                ids.append(str(sid))
    return ids


async def _paginate_channel_videos(kind: str, uid: int, sid: int, credential: Credential | None, timeout: float = 30.0, ps: int = 100, **_kw: Any) -> dict[str, Any]:
    u = user.User(uid, credential=credential)
    all_archives: list[Any] = []
    pn = 1
    while True:
        async with _map_bilibili_errors(f"channel_videos_{kind}[{sid}]"):
            if kind == "season":
                data = await asyncio.wait_for(
                    u.get_channel_videos_season(sid=sid, sort=ChannelOrder.DEFAULT, pn=pn, ps=ps),
                    timeout=timeout,
                )
            else:
                data = await asyncio.wait_for(
                    u.get_channel_videos_series(sid=sid, sort=ChannelOrder.DEFAULT, pn=pn, ps=ps),
                    timeout=timeout,
                )
        archives = data.get("archives", [])
        all_archives.extend(archives)
        page_info = data.get("page", {})
        total = page_info.get("count", 0)
        if not archives or (total > 0 and pn * ps >= total):
            break
        pn += 1
    return {"archives": all_archives, "page": {"count": len(all_archives)}}


# ---------------------------------------------------------------------------
# User methods
# ---------------------------------------------------------------------------

async def fetch_user_channels(uid: int, cred: Credential | None = None, timeout: float = 30.0, **_kw: Any) -> dict[str, Any]:
    u = user.User(uid, credential=cred)
    async with _map_bilibili_errors("channels"):
        channels = await asyncio.wait_for(u.get_channels(), timeout=timeout)
    rows: list[dict[str, Any]] = []
    for ch in channels:
        try:
            ch_type = ch.get_type()
            ch_id = ch.get_id()
            async with _map_bilibili_errors(f"channels[{ch_id}]: get_meta"):
                meta = await asyncio.wait_for(ch.get_meta(), timeout=timeout)
            rows.append({"id": ch_id, "type": json_safe(ch_type), "meta": json_safe(meta)})
        except Exception as exc:
            raise _RequestError(f"channels: serialise failed: {exc}") from exc
    return {"channels": rows}


async def fetch_user_media_list(uid: int, cred: Credential | None = None, timeout: float = 30.0, sort_field: int | user.MedialistOrder = user.MedialistOrder.PUBDATE, **kw: Any) -> dict[str, Any]:
    if isinstance(sort_field, int):
        try:
            sort_enum = user.MedialistOrder(sort_field)
        except ValueError as exc:
            raise _InvalidRequestError(f"media_list: invalid sort_field {sort_field!r}: {exc}") from exc
    else:
        sort_enum = sort_field
    u = user.User(uid, credential=cred)
    async with _map_bilibili_errors("media_list"):
        return await asyncio.wait_for(u.get_media_list(sort_field=sort_enum, **kw), timeout=timeout)


def _extract_qa_ids_from_upower_qa(raw_payload: dict) -> list[str]:
    ids: list[str] = []
    for page in raw_payload.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        for item in page.get("list", []) or []:
            if not isinstance(item, dict):
                continue
            qa_id = item.get("qa_id")
            if qa_id is not None:
                ids.append(str(qa_id))
    return ids


async def fetch_upower_qa_detail_item(qa_id: str, credential: Credential | None, timeout: float = 30.0, **kw: Any) -> dict[str, Any]:
    try:
        qa_id_int = int(qa_id)
        uid = int(kw["_uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _InvalidRequestError(f"upower_qa_detail[{qa_id}]: invalid input: {exc}") from exc
    u = user.User(uid, credential=credential)
    async with _map_bilibili_errors(f"upower_qa_detail[{qa_id}]"):
        result = await asyncio.wait_for(u.get_upower_qa_detail(qa_id_int), timeout=timeout)
    return normalise_api_result(result, key="detail")


# ---------------------------------------------------------------------------
# Login helpers
# ---------------------------------------------------------------------------

async def login_qr_generate() -> dict[str, Any]:
    from bilibili_api import login_v2
    loop = asyncio.get_running_loop()
    credential, qrcode_key, qrcode_url = await loop.run_in_executor(None, login_v2.generate_qrcode)
    return {"qrcode_key": qrcode_key, "qrcode_url": qrcode_url}


async def login_qr_check(qrcode_key: str) -> dict[str, Any]:
    from bilibili_api import login_v2
    loop = asyncio.get_running_loop()
    try:
        credential, status, msg = await loop.run_in_executor(None, login_v2.check_qrcode_events, qrcode_key)
        result: dict[str, Any] = {"status": status, "message": msg}
        if credential is not None:
            result["sessdata"] = credential.sessdata
            result["bili_jct"] = credential.bili_jct
            result["buvid3"] = credential.buvid3
        return result
    except Exception as exc:
        return {"status": "ERROR", "message": str(exc)}


# ---------------------------------------------------------------------------
# Exports referenced by catalog.py
# ---------------------------------------------------------------------------

__all__ = [
    "_extract_bvids_from_videos",
    "_extract_cvids_from_articles",
    "_extract_opus_ids_from_opus",
    "_extract_qa_ids_from_upower_qa",
    "_extract_rlids_from_article_list",
    "_extract_season_ids",
    "_extract_series_ids",
    "_paginate_channel_videos",
    "_user_method",
    "_wrap_list_result",
    "_wrap_scalar_result",
    "fetch_article_detail_item",
    "fetch_article_list_detail_item",
    "fetch_endpoint",
    "fetch_item",
    "fetch_opus_detail_item",
    "fetch_upower_qa_detail_item",
    "fetch_user_channels",
    "fetch_user_media_list",
    "fetch_video_ai_conclusion_item",
    "fetch_video_chargers_item",
    "fetch_video_danmaku_snapshot_item",
    "fetch_video_danmaku_view_item",
    "fetch_video_danmaku_xml_item",
    "fetch_video_danmakus_item",
    "fetch_video_detail_full_item",
    "fetch_video_detail_item",
    "fetch_video_download_url_item",
    "fetch_video_is_episode_item",
    "fetch_video_is_forbid_note_item",
    "fetch_video_online_item",
    "fetch_video_pages_item",
    "fetch_video_pay_coins_item",
    "fetch_video_pbp_item",
    "fetch_video_player_info_item",
    "fetch_video_private_notes_item",
    "fetch_video_public_notes_item",
    "fetch_video_related_item",
    "fetch_video_relation_item",
    "fetch_video_snapshot_item",
    "fetch_video_special_dms_item",
    "fetch_video_subtitle_item",
    "fetch_video_up_mid_item",
    "init_http_backend",
    "json_safe",
    "login_qr_check",
    "login_qr_generate",
    "map_internal_exception",
    "resolve_audio_url",
]
