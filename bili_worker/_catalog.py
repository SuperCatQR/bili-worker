"""Worker-side SDK callable catalog (contract §3, §6).

This module is the **only** place that defines the actual SDK callables.
Every ``EndpointSpec.callable`` from the main process is reified here as a
named entry in ``CATALOG``. The op dispatch loop routes ``fetch_page`` /
``fetch_item`` by endpoint name to the corresponding entry.

The main process receives only the serializable metadata via
``describe_catalog`` — callable functions never cross the IPC boundary.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from bilibili_api import Credential, request_settings, select_client, user
from bilibili_api.article import Article, ArticleList
from bilibili_api.channel_series import ChannelOrder
from bilibili_api.exceptions import ApiException, InitialStateException
from bilibili_api.opus import Opus
from bilibili_api.video import AudioQuality, Video, VideoDownloadURLDataDetecter

logger = logging.getLogger("bili_worker.catalog")

# ---------------------------------------------------------------------------
# Catalog entry type
# ---------------------------------------------------------------------------

PaginationStrategy = str  # "none" | "page" | "cursor" | "anchor" | "legacy_offset" | "oid" | "custom"


@dataclass
class CatalogEntry:
    """Serializable metadata + local callable for one endpoint."""

    name: str
    kind: str  # "uid" | "item"
    callable: Callable[..., Awaitable[dict]]
    credential_required: bool = False
    params_strategy: dict[str, Any] = field(default_factory=dict)
    pagination_strategy: PaginationStrategy = "none"
    rate_limit_key: str = ""
    item_id_path: str | None = None
    item_id_paths: list[str] | None = None
    items_path: str | None = None
    source_endpoint: str | None = None
    needs_parent_uid: bool = False


# ---------------------------------------------------------------------------
# Helpers (ported from _bilibili_adapter / _adapter_core)
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Convert bilibili-api return objects into JSON-serialisable values."""
    from enum import Enum

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _json_safe(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _normalise_api_result(result: Any, key: str = "data") -> dict[str, Any]:
    safe = _json_safe(result)
    if isinstance(safe, dict):
        return safe
    return {key: safe}


def _user_method(name: str, **defaults: Any):
    """Build a uid-level callable that dispatches to ``User.{name}``."""

    async def _fn(uid, cred=None, **kw):
        merged = {**defaults, **kw}
        return await getattr(user.User(uid, credential=cred), name)(**merged)

    return _fn


async def _wrap_scalar_result(coro, key: str = "value") -> dict:
    result = await coro
    safe = _json_safe(result)
    if isinstance(safe, dict):
        return safe
    return {key: safe}


async def _wrap_list_result(coro) -> dict:
    return _normalise_api_result(await coro, key="list")


# ---------------------------------------------------------------------------
# Video adapter callables (ported from _adapters/_video.py)
# ---------------------------------------------------------------------------

async def _fetch_video_detail_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    info = await asyncio.wait_for(v.get_info(), timeout=timeout)
    return _json_safe(info)


async def _fetch_video_pages_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    pages = await asyncio.wait_for(v.get_pages(), timeout=timeout)
    return _json_safe(pages)


async def _fetch_video_detail_full_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    detail = await asyncio.wait_for(v.get_detail(), timeout=timeout)
    return _json_safe(detail)


async def _fetch_video_ai_conclusion_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_ai_conclusion(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_danmaku_snapshot_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_danmaku_snapshot(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_danmaku_view_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_danmaku_view(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_danmaku_xml_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_danmaku_xml(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_danmakus_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_danmakus(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_online_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_online(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_pay_coins_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_pay_coins(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_pbp_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_pbp(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_player_info_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_player_info(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_private_notes_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_private_notes(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_public_notes_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_public_notes(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_related_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_related(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_relation_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_relation(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_special_dms_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_special_dms(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_subtitle_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    """Fetch video subtitle info (contract §10: aiohttp in worker, small text)."""
    import aiohttp

    v = Video(bvid=bvid, credential=credential)
    info = await asyncio.wait_for(v.get_subtitle(), timeout=timeout)
    subtitles = info.get("subtitles", []) if isinstance(info, dict) else []

    results: list[dict[str, Any]] = []
    for sub in subtitles:
        sub_url = sub.get("subtitle_url", "") if isinstance(sub, dict) else ""
        if not sub_url:
            continue
        if sub_url.startswith("//"):
            sub_url = "https:" + sub_url
        try:
            async with aiohttp.ClientSession() as session, session.get(sub_url) as resp:
                body = await resp.json()
        except Exception:
            body = None
        results.append({"meta": sub, "body": body})

    return {"subtitles": results}


async def _fetch_video_up_mid_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_up_mid(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_snapshot_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_snapshot(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_download_url_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_download_url(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_is_episode_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.is_episode(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_is_forbid_note_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.is_forbid_note(), timeout=timeout)
    return _json_safe(result)


async def _fetch_video_chargers_item(
    bvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    v = Video(bvid=bvid, credential=credential)
    result = await asyncio.wait_for(v.get_chargers(), timeout=timeout)
    return _json_safe(result)


# ---------------------------------------------------------------------------
# Content callables (ported from _bilibili_adapter.py)
# ---------------------------------------------------------------------------

async def _fetch_article_detail_item(
    cvid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:

    try:
        cvid_int = int(cvid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"article_detail[{cvid}]: invalid cvid: {exc}") from exc

    a = Article(cvid_int, credential=credential)
    info = await asyncio.wait_for(a.get_info(), timeout=timeout)

    try:
        await asyncio.wait_for(a.fetch_content(), timeout=timeout)
        markdown_text: str = a.markdown()
        content_json: list[Any] = a.json()
    except InitialStateException as exc:
        raise ValueError(
            f"article_detail[{cvid}]: fetch_content {exc} (article unavailable)",
        ) from exc
    except KeyError as exc:
        raise ValueError(
            f"article_detail[{cvid}]: fetch_content missing key {exc} (article unavailable)",
        ) from exc

    return {"info": info, "markdown": markdown_text, "content_json": content_json}


async def _fetch_opus_detail_item(
    opus_id: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    try:
        opus_id_int = int(opus_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"opus_detail[{opus_id}]: invalid opus_id: {exc}") from exc

    o = Opus(opus_id_int, credential=credential)

    try:
        info = await asyncio.wait_for(o.get_info(), timeout=timeout)
    except ApiException as exc:
        if "opus_id 不正确" in str(exc) or "fallback" in str(exc).lower():
            raise ValueError(f"opus_detail[{opus_id}]: opus unavailable ({exc})") from exc
        raise

    try:
        markdown_text: str = await asyncio.wait_for(o.markdown(), timeout=timeout)
        images: list[dict[str, Any]] = await asyncio.wait_for(o.get_images_raw_info(), timeout=timeout)
    except KeyError as exc:
        raise ValueError(
            f"opus_detail[{opus_id}]: markdown missing key {exc} (opus unavailable)",
        ) from exc

    return {"info": info, "markdown": markdown_text, "images": images}


async def _fetch_article_list_detail_item(
    rlid: str, credential: Credential | None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    try:
        rlid_int = int(rlid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"article_list_detail[{rlid}]: invalid rlid: {exc}") from exc

    al = ArticleList(rlid_int, credential=credential)
    result = await asyncio.wait_for(al.get_content(), timeout=timeout)
    return result


# ---------------------------------------------------------------------------
# Channel / upower callables
# ---------------------------------------------------------------------------

async def _fetch_user_channels(
    uid: int, cred: Credential | None = None, timeout: float = 30.0, **_kw: Any,
) -> dict[str, Any]:
    u = user.User(uid, credential=cred)
    channels = await asyncio.wait_for(u.get_channels(), timeout=timeout)

    rows: list[dict[str, Any]] = []
    for ch in channels:
        try:
            ch_type = ch.get_type()
            ch_id = ch.get_id()
            meta = await asyncio.wait_for(ch.get_meta(), timeout=timeout)
            rows.append({
                "id": ch_id,
                "type": _json_safe(ch_type),
                "meta": _json_safe(meta),
            })
        except Exception as exc:
            raise ValueError(f"channels: serialise failed: {exc}") from exc
    return {"channels": rows}


async def _fetch_user_media_list(
    uid: int, cred: Credential | None = None, timeout: float = 30.0,
    sort_field: int = 2, **kw: Any,
) -> dict[str, Any]:
    if isinstance(sort_field, int):
        try:
            sort_enum = user.MedialistOrder(sort_field)
        except ValueError as exc:
            raise ValueError(f"media_list: invalid sort_field {sort_field!r}: {exc}") from exc
    else:
        sort_enum = sort_field

    u = user.User(uid, credential=cred)
    return await asyncio.wait_for(
        u.get_media_list(sort_field=sort_enum, **kw), timeout=timeout,
    )


async def _paginate_channel_videos(
    kind: str, uid: int, sid: int, credential: Credential | None,
    timeout: float = 30.0, ps: int = 100, **_kw: Any,
) -> dict[str, Any]:
    u = user.User(uid, credential=credential)
    all_archives: list[Any] = []
    pn = 1
    while True:
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


async def _fetch_upower_qa_detail_item(
    qa_id: str, credential: Credential | None, timeout: float = 30.0, **kw: Any,
) -> dict[str, Any]:
    try:
        qa_id_int = int(qa_id)
        uid = int(kw["_uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"upower_qa_detail[{qa_id}]: invalid input: {exc}") from exc

    u = user.User(uid, credential=credential)
    result = await asyncio.wait_for(u.get_upower_qa_detail(qa_id_int), timeout=timeout)
    return _normalise_api_result(result, key="detail")


# ---------------------------------------------------------------------------
# Audio URL resolution (contract §6.4, §10)
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
    for key in ("dash", "data"):
        inner = data.get(key)
        if isinstance(inner, dict):
            dur = inner.get("duration")
            if dur is not None:
                return float(dur) / 1000.0
    dur = data.get("duration")
    if dur is not None:
        return float(dur)
    return None


async def _resolve_audio_url(
    bvid: str, page_index: int, quality: str, credential: Credential | None,
) -> dict[str, Any]:
    video = Video(bvid=bvid, credential=credential)
    try:
        data = await video.get_download_url(page_index=page_index)
    except Exception as exc:
        raise ValueError(f"get_download_url failed for {bvid}: {exc}") from exc

    detecter = VideoDownloadURLDataDetecter(data)
    quality_enum = _resolve_quality(quality)
    streams = detecter.detect(audio_max_quality=quality_enum)

    audio_stream = None
    for s in streams:
        if type(s).__name__ == "AudioStreamDownloadURL":
            audio_stream = s
            break

    if audio_stream is None:
        raise ValueError(f"no audio stream found for {bvid} page {page_index}")

    duration = _extract_duration(data)
    return {
        "url": audio_stream.url,
        "quality": str(getattr(audio_stream, "audio_quality", quality)),
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# HTTP backend init
# ---------------------------------------------------------------------------

def _init_http_backend(backend: str = "aiohttp", impersonate: str = "chrome131") -> None:
    try:
        import curl_cffi  # noqa: F401
        if backend == "curl_cffi":
            select_client("curl_cffi")
            request_settings.set("impersonate", impersonate)
            logger.info("HTTP backend: curl_cffi (impersonate=%s)", impersonate)
            return
    except ImportError:
        if backend == "curl_cffi":
            logger.warning("curl_cffi not installed; falling back to aiohttp")
    select_client("aiohttp")
    logger.info("HTTP backend: aiohttp")


# ---------------------------------------------------------------------------
# Catalog — all 63 endpoints
# ---------------------------------------------------------------------------

CATALOG: list[CatalogEntry] = [
    # ===== user_endpoints (uid-level) =====
    CatalogEntry(name="user_info", kind="uid", callable=_user_method("get_user_info"),
                 pagination_strategy="none", rate_limit_key="user_info"),
    CatalogEntry(name="videos", kind="uid",
                 callable=_user_method("get_videos", pn=1, ps=30, tid=0, keyword="", order=user.VideoOrder.PUBDATE),
                 params_strategy={"pn": 1, "ps": 30}, pagination_strategy="page", rate_limit_key="videos",
                 item_id_path="list.vlist[*].bvid", items_path="list.vlist"),
    CatalogEntry(name="access_id", kind="uid",
                 callable=lambda uid, cred=None, **kw: _wrap_scalar_result(
                     user.User(uid, credential=cred).get_access_id(), key="access_id"),
                 pagination_strategy="none", rate_limit_key="access_id"),
    CatalogEntry(name="relation_info", kind="uid", callable=_user_method("get_relation_info"),
                 pagination_strategy="none", rate_limit_key="relation_info"),
    CatalogEntry(name="up_stat", kind="uid", callable=_user_method("get_up_stat"),
                 pagination_strategy="none", rate_limit_key="up_stat"),
    CatalogEntry(name="overview_stat", kind="uid", callable=_user_method("get_overview_stat"),
                 pagination_strategy="none", rate_limit_key="overview_stat"),
    CatalogEntry(name="articles", kind="uid",
                 callable=_user_method("get_articles", pn=1, ps=30, order=user.ArticleOrder.PUBDATE),
                 params_strategy={"pn": 1, "ps": 30}, pagination_strategy="page", rate_limit_key="articles",
                 item_id_path="articles[*].id", items_path="articles"),
    CatalogEntry(name="subscribed_bangumi", kind="uid",
                 callable=_user_method("get_subscribed_bangumi", pn=1, ps=15, type_=user.BangumiType.BANGUMI,
                                       follow_status=user.BangumiFollowStatus.ALL),
                 params_strategy={"pn": 1, "ps": 15}, pagination_strategy="page", rate_limit_key="subscribed_bangumi",
                 item_id_path="list[*].season_id", items_path="list"),
    CatalogEntry(name="opus", kind="uid",
                 callable=_user_method("get_opus", type_=user.OpusType.ALL, offset=""),
                 params_strategy={"offset": ""}, pagination_strategy="cursor", rate_limit_key="opus",
                 item_id_path="items[*].opus_id", items_path="items"),
    CatalogEntry(name="dynamics", kind="uid",
                 callable=_user_method("get_dynamics_new", offset=""),
                 params_strategy={"offset": ""}, pagination_strategy="cursor", rate_limit_key="dynamics",
                 item_id_path="items[*].id_str", items_path="items"),
    CatalogEntry(name="audios", kind="uid",
                 callable=_user_method("get_audios", pn=1, ps=30, order=user.AudioOrder.PUBDATE),
                 params_strategy={"pn": 1, "ps": 30}, pagination_strategy="page", rate_limit_key="audios",
                 item_id_path="data[*].id", items_path="data"),
    CatalogEntry(name="channel_list", kind="uid",
                 callable=_user_method("get_channel_list", pn=1, ps=20),
                 params_strategy={"pn": 1, "ps": 20}, pagination_strategy="page", rate_limit_key="channel_list",
                 item_id_paths=["items_lists.seasons_list[*].meta.season_id",
                                "items_lists.series_list[*].meta.series_id"],
                 items_path="items_lists"),
    CatalogEntry(name="channels", kind="uid", callable=_fetch_user_channels,
                 pagination_strategy="none", rate_limit_key="channels"),
    CatalogEntry(name="media_list", kind="uid", callable=_fetch_user_media_list,
                 params_strategy={"oid": None, "ps": 100, "direction": True, "desc": True,
                                  "sort_field": 2, "tid": 0, "with_current": False},
                 pagination_strategy="oid", rate_limit_key="media_list",
                 item_id_paths=["media_list[*].bvid", "list[*].bvid", "items[*].bvid"],
                 items_path="media_list"),

    # ===== video_endpoints (item-level) =====
    CatalogEntry(name="video_detail", kind="item", callable=_fetch_video_detail_item,
                 pagination_strategy="none", rate_limit_key="video_detail",
                 source_endpoint="videos"),
    CatalogEntry(name="video_pages", kind="item", callable=_fetch_video_pages_item,
                 pagination_strategy="none", rate_limit_key="video_pages",
                 source_endpoint="videos"),
    CatalogEntry(name="video_detail_full", kind="item", callable=_fetch_video_detail_full_item,
                 pagination_strategy="none", rate_limit_key="video_detail_full",
                 source_endpoint="videos"),
    CatalogEntry(name="video_ai_conclusion", kind="item", callable=_fetch_video_ai_conclusion_item,
                 pagination_strategy="none", rate_limit_key="video_ai_conclusion",
                 source_endpoint="videos"),
    CatalogEntry(name="video_danmaku_snapshot", kind="item", callable=_fetch_video_danmaku_snapshot_item,
                 pagination_strategy="none", rate_limit_key="video_danmaku_snapshot",
                 source_endpoint="videos"),
    CatalogEntry(name="video_danmaku_view", kind="item", callable=_fetch_video_danmaku_view_item,
                 pagination_strategy="none", rate_limit_key="video_danmaku_view",
                 source_endpoint="videos"),
    CatalogEntry(name="video_danmaku_xml", kind="item", callable=_fetch_video_danmaku_xml_item,
                 pagination_strategy="none", rate_limit_key="video_danmaku_xml",
                 source_endpoint="videos"),
    CatalogEntry(name="video_danmakus", kind="item", callable=_fetch_video_danmakus_item,
                 pagination_strategy="none", rate_limit_key="video_danmakus",
                 source_endpoint="videos"),
    CatalogEntry(name="video_online", kind="item", callable=_fetch_video_online_item,
                 pagination_strategy="none", rate_limit_key="video_online",
                 source_endpoint="videos"),
    CatalogEntry(name="video_pay_coins", kind="item", callable=_fetch_video_pay_coins_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_pay_coins",
                 source_endpoint="videos"),
    CatalogEntry(name="video_pbp", kind="item", callable=_fetch_video_pbp_item,
                 pagination_strategy="none", rate_limit_key="video_pbp",
                 source_endpoint="videos"),
    CatalogEntry(name="video_player_info", kind="item", callable=_fetch_video_player_info_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_player_info",
                 source_endpoint="videos"),
    CatalogEntry(name="video_private_notes", kind="item", callable=_fetch_video_private_notes_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_private_notes",
                 source_endpoint="videos"),
    CatalogEntry(name="video_public_notes", kind="item", callable=_fetch_video_public_notes_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_public_notes",
                 source_endpoint="videos"),
    CatalogEntry(name="video_related", kind="item", callable=_fetch_video_related_item,
                 pagination_strategy="none", rate_limit_key="video_related",
                 source_endpoint="videos"),
    CatalogEntry(name="video_relation", kind="item", callable=_fetch_video_relation_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_relation",
                 source_endpoint="videos"),
    CatalogEntry(name="video_special_dms", kind="item", callable=_fetch_video_special_dms_item,
                 pagination_strategy="none", rate_limit_key="video_special_dms",
                 source_endpoint="videos"),
    CatalogEntry(name="video_subtitle", kind="item", callable=_fetch_video_subtitle_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_subtitle",
                 source_endpoint="videos"),
    CatalogEntry(name="video_up_mid", kind="item", callable=_fetch_video_up_mid_item,
                 pagination_strategy="none", rate_limit_key="video_up_mid",
                 source_endpoint="videos"),
    CatalogEntry(name="video_snapshot", kind="item", callable=_fetch_video_snapshot_item,
                 pagination_strategy="none", rate_limit_key="video_snapshot",
                 source_endpoint="videos"),
    CatalogEntry(name="video_download_url", kind="item", callable=_fetch_video_download_url_item,
                 pagination_strategy="none", rate_limit_key="video_download_url",
                 source_endpoint="videos"),
    CatalogEntry(name="video_is_episode", kind="item", callable=_fetch_video_is_episode_item,
                 pagination_strategy="none", rate_limit_key="video_is_episode",
                 source_endpoint="videos"),
    CatalogEntry(name="video_is_forbid_note", kind="item", callable=_fetch_video_is_forbid_note_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="video_is_forbid_note",
                 source_endpoint="videos"),
    CatalogEntry(name="video_chargers", kind="item", callable=_fetch_video_chargers_item,
                 pagination_strategy="none", rate_limit_key="video_chargers",
                 source_endpoint="videos"),

    # ===== content_endpoints (item-level) =====
    CatalogEntry(name="article_detail", kind="item", callable=_fetch_article_detail_item,
                 pagination_strategy="none", rate_limit_key="article_detail",
                 source_endpoint="articles"),
    CatalogEntry(name="opus_detail", kind="item", callable=_fetch_opus_detail_item,
                 pagination_strategy="none", rate_limit_key="opus_detail",
                 source_endpoint="opus"),
    CatalogEntry(name="article_list_detail", kind="item", callable=_fetch_article_list_detail_item,
                 pagination_strategy="none", rate_limit_key="article_list_detail",
                 source_endpoint="article_list"),

    # ===== channel_and_upower_endpoints =====
    CatalogEntry(name="user_medal", kind="uid", callable=_user_method("get_user_medal"),
                 credential_required=True, pagination_strategy="none", rate_limit_key="user_medal"),
    CatalogEntry(name="live_info", kind="uid", callable=_user_method("get_live_info"),
                 pagination_strategy="none", rate_limit_key="live_info"),
    CatalogEntry(name="user_relation", kind="uid", callable=_user_method("get_relation"),
                 credential_required=True, pagination_strategy="none", rate_limit_key="user_relation"),
    CatalogEntry(name="reservation", kind="uid", callable=_user_method("get_reservation"),
                 pagination_strategy="none", rate_limit_key="reservation"),
    CatalogEntry(name="uplikeimg", kind="uid", callable=_user_method("get_uplikeimg"),
                 credential_required=True, pagination_strategy="none", rate_limit_key="uplikeimg"),
    CatalogEntry(name="top_followers", kind="uid", callable=_user_method("top_followers", since=None),
                 pagination_strategy="none", rate_limit_key="top_followers"),
    CatalogEntry(name="space_notice", kind="uid", callable=_user_method("get_space_notice"),
                 pagination_strategy="none", rate_limit_key="space_notice"),
    CatalogEntry(name="all_followings", kind="uid", callable=_user_method("get_all_followings"),
                 credential_required=True, pagination_strategy="none", rate_limit_key="all_followings"),
    CatalogEntry(name="followings", kind="uid",
                 callable=_user_method("get_followings", pn=1, ps=100, attention=False, order=user.OrderType.desc),
                 credential_required=True, params_strategy={"pn": 1, "ps": 100},
                 pagination_strategy="page", rate_limit_key="followings",
                 item_id_path="list[*].mid", items_path="list"),
    CatalogEntry(name="followers", kind="uid",
                 callable=_user_method("get_followers", pn=1, ps=100, desc=True),
                 credential_required=True, params_strategy={"pn": 1, "ps": 100},
                 pagination_strategy="page", rate_limit_key="followers",
                 item_id_path="list[*].mid", items_path="list"),
    CatalogEntry(name="same_followers", kind="uid",
                 callable=_user_method("get_self_same_followers", pn=1, ps=50),
                 credential_required=True, params_strategy={"pn": 1, "ps": 50},
                 pagination_strategy="page", rate_limit_key="same_followers",
                 item_id_path="list[*].mid", items_path="list"),
    CatalogEntry(name="top_videos", kind="uid", callable=_user_method("get_top_videos"),
                 pagination_strategy="none", rate_limit_key="top_videos"),
    CatalogEntry(name="masterpiece", kind="uid",
                 callable=lambda uid, cred=None, **kw: _wrap_list_result(
                     user.User(uid, credential=cred).get_masterpiece()),
                 pagination_strategy="none", rate_limit_key="masterpiece"),
    CatalogEntry(name="article_list", kind="uid",
                 callable=_user_method("get_article_list", order=user.ArticleListOrder.LATEST),
                 pagination_strategy="none", rate_limit_key="article_list"),
    CatalogEntry(name="cheese", kind="uid", callable=_user_method("get_cheese"),
                 pagination_strategy="none", rate_limit_key="cheese"),
    CatalogEntry(name="elec_monthly", kind="uid", callable=_user_method("get_elec_user_monthly"),
                 credential_required=True, pagination_strategy="none", rate_limit_key="elec_monthly"),
    CatalogEntry(name="user_fav_tag", kind="uid",
                 callable=_user_method("get_user_fav_tag", pn=1, ps=20),
                 params_strategy={"pn": 1, "ps": 20}, pagination_strategy="page", rate_limit_key="user_fav_tag"),
    CatalogEntry(name="album", kind="uid",
                 callable=lambda uid, cred=None, **kw: user.User(uid, credential=cred).get_album(
                     biz=kw.get("biz", user.AlbumType.ALL),
                     page_num=kw.get("pn", 1), page_size=kw.get("ps", 30)),
                 params_strategy={"pn": 1, "ps": 30}, pagination_strategy="page", rate_limit_key="album",
                 items_path="biz_list"),
    CatalogEntry(name="channel_videos_season", kind="item",
                 callable=lambda sid, cred=None, **kw: _paginate_channel_videos(
                     "season", kw["_uid"], int(sid), cred),
                 pagination_strategy="none", rate_limit_key="channel_videos_season",
                 source_endpoint="channel_list", needs_parent_uid=True),
    CatalogEntry(name="channel_videos_series", kind="item",
                 callable=lambda sid, cred=None, **kw: _paginate_channel_videos(
                     "series", kw["_uid"], int(sid), cred),
                 pagination_strategy="none", rate_limit_key="channel_videos_series",
                 source_endpoint="channel_list", needs_parent_uid=True),
    CatalogEntry(name="upower_qa", kind="uid",
                 callable=_user_method("get_upower_qa_list", anchor=0),
                 credential_required=True, params_strategy={"anchor": 0},
                 pagination_strategy="anchor", rate_limit_key="upower_qa",
                 item_id_path="list[*].qa_id", items_path="list"),
    CatalogEntry(name="upower_qa_detail", kind="item", callable=_fetch_upower_qa_detail_item,
                 credential_required=True, pagination_strategy="none", rate_limit_key="upower_qa_detail",
                 source_endpoint="upower_qa", needs_parent_uid=True),
]

CATALOG_BY_NAME: dict[str, CatalogEntry] = {ep.name: ep for ep in CATALOG}


# ---------------------------------------------------------------------------
# Op dispatch helpers (called from __main__)
# ---------------------------------------------------------------------------

async def _call_page(
    uid: int, endpoint: str, credential: Credential | None,
    request_params: dict[str, Any], timeout_s: float,
) -> dict[str, Any]:
    """Execute a uid-level endpoint callable (contract §6.3 fetch_page)."""
    entry = CATALOG_BY_NAME.get(endpoint)
    if entry is None:
        raise ValueError(f"unknown endpoint: {endpoint!r}")
    if entry.kind != "uid":
        raise ValueError(f"endpoint {endpoint!r} is kind={entry.kind}, not uid")

    result = await asyncio.wait_for(
        entry.callable(uid, cred=credential, **request_params),
        timeout=timeout_s,
    )
    return _json_safe(result)


async def _call_item(
    endpoint: str, item_id: str, credential: Credential | None,
    parent_uid: int | None, timeout_s: float,
) -> dict[str, Any]:
    """Execute an item-level endpoint callable (contract §6.3 fetch_item)."""
    entry = CATALOG_BY_NAME.get(endpoint)
    if entry is None:
        raise ValueError(f"unknown endpoint: {endpoint!r}")
    if entry.kind != "item":
        raise ValueError(f"endpoint {endpoint!r} is kind={entry.kind}, not item")

    extra: dict[str, Any] = {"timeout": timeout_s}
    if entry.needs_parent_uid and parent_uid is not None:
        extra["_uid"] = parent_uid

    result = await asyncio.wait_for(
        entry.callable(item_id, credential, **extra),
        timeout=timeout_s,
    )
    return _json_safe(result)


__all__ = [
    "CATALOG",
    "CATALOG_BY_NAME",
    "CatalogEntry",
    "_call_item",
    "_call_page",
    "_init_http_backend",
    "_resolve_audio_url",
]
