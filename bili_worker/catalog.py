"""Endpoint catalog — port of bili_unit.fetching._endpoint_catalog + _endpoint_groups.

This module lives in the worker (the ONLY place that imports bilibili_api).
The main process receives the catalog via ``describe_catalog`` and never
imports this module (arm's-length boundary, contract §12).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from bilibili_api import user
from bilibili_api.article import Article, ArticleList
from bilibili_api.channel_series import ChannelOrder
from bilibili_api.opus import Opus
from bilibili_api.video import Video

from .sdk_adapter import (
    _extract_bvids_from_videos,
    _extract_cvids_from_articles,
    _extract_opus_ids_from_opus,
    _extract_qa_ids_from_upower_qa,
    _extract_rlids_from_article_list,
    _extract_season_ids,
    _extract_series_ids,
    _paginate_channel_videos,
    _user_method,
    _wrap_list_result,
    _wrap_scalar_result,
    fetch_article_detail_item,
    fetch_article_list_detail_item,
    fetch_opus_detail_item,
    fetch_upower_qa_detail_item,
    fetch_user_channels,
    fetch_user_media_list,
    fetch_video_ai_conclusion_item,
    fetch_video_chargers_item,
    fetch_video_danmaku_snapshot_item,
    fetch_video_danmaku_view_item,
    fetch_video_danmaku_xml_item,
    fetch_video_danmakus_item,
    fetch_video_detail_full_item,
    fetch_video_detail_item,
    fetch_video_download_url_item,
    fetch_video_is_episode_item,
    fetch_video_is_forbid_note_item,
    fetch_video_online_item,
    fetch_video_pages_item,
    fetch_video_pay_coins_item,
    fetch_video_pbp_item,
    fetch_video_player_info_item,
    fetch_video_private_notes_item,
    fetch_video_public_notes_item,
    fetch_video_related_item,
    fetch_video_relation_item,
    fetch_video_snapshot_item,
    fetch_video_special_dms_item,
    fetch_video_subtitle_item,
    fetch_video_up_mid_item,
)

# ---------------------------------------------------------------------------
# EndpointSpec (worker-side mirror of bili_unit.fetching._endpoint_spec)
# ---------------------------------------------------------------------------

PaginationStrategy = str


@dataclass
class EndpointSpec:
    name: str
    callable: Callable[..., Awaitable[dict]]
    credential_required: bool = False
    params_strategy: dict[str, Any] = field(default_factory=dict)
    pagination_strategy: PaginationStrategy = "none"
    rate_limit_key: str = ""
    item_id_path: str | None = None
    item_id_paths: list[str] | None = None
    items_path: str | None = None
    kind: str = "uid"
    source_endpoint: str | None = None
    extract_items: Callable[[dict], list[str]] | None = None
    skip_item: Callable[[dict], str | None] | None = None
    needs_parent_uid: bool = False


# ---------------------------------------------------------------------------
# Endpoint groups
# ---------------------------------------------------------------------------

def _skip_legacy_article_detail_item(item: dict) -> str | None:
    try:
        template_id = item.get("template_id")
        origin_template_id = item.get("origin_template_id")
        category = item.get("category") or {}
        category_id = category.get("id")
        type_id = item.get("type")
    except AttributeError:
        return None
    if (
        template_id == 4
        and origin_template_id == 5
        and category_id in {41, 42}
        and type_id in {2, 3, 4, 0}
    ):
        return "legacy article body endpoint skips note/opus-style content"
    return None


def user_endpoints() -> list[EndpointSpec]:
    return [
        EndpointSpec(
            name="user_info",
            callable=_user_method("get_user_info"),
            credential_required=False,
            pagination_strategy="none",
            rate_limit_key="user_info",
        ),
        EndpointSpec(
            name="videos",
            callable=_user_method("get_videos", pn=1, ps=30, tid=0, keyword="", order=user.VideoOrder.PUBDATE),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 30},
            pagination_strategy="page",
            rate_limit_key="videos",
            item_id_path="list.vlist[*].bvid",
            items_path="list.vlist",
        ),
        EndpointSpec(
            name="access_id",
            callable=lambda uid, cred=None, **kw: (
                _wrap_scalar_result(
                    user.User(uid, credential=cred).get_access_id(),
                    key="access_id",
                )
            ),
            credential_required=False,
            pagination_strategy="none",
            rate_limit_key="access_id",
        ),
        EndpointSpec(
            name="relation_info",
            callable=_user_method("get_relation_info"),
            credential_required=False,
            pagination_strategy="none",
            rate_limit_key="relation_info",
        ),
        EndpointSpec(
            name="up_stat",
            callable=_user_method("get_up_stat"),
            credential_required=False,
            pagination_strategy="none",
            rate_limit_key="up_stat",
        ),
        EndpointSpec(
            name="overview_stat",
            callable=_user_method("get_overview_stat"),
            credential_required=False,
            pagination_strategy="none",
            rate_limit_key="overview_stat",
        ),
        EndpointSpec(
            name="articles",
            callable=_user_method("get_articles", pn=1, ps=30, order=user.ArticleOrder.PUBDATE),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 30},
            pagination_strategy="page",
            rate_limit_key="articles",
            item_id_path="articles[*].id",
            items_path="articles",
        ),
        EndpointSpec(
            name="subscribed_bangumi",
            callable=_user_method(
                "get_subscribed_bangumi",
                pn=1, ps=15, type_=user.BangumiType.BANGUMI,
                follow_status=user.BangumiFollowStatus.ALL,
            ),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 15},
            pagination_strategy="page",
            rate_limit_key="subscribed_bangumi",
            item_id_path="list[*].season_id",
            items_path="list",
        ),
        EndpointSpec(
            name="opus",
            callable=_user_method("get_opus", type_=user.OpusType.ALL, offset=""),
            credential_required=False,
            params_strategy={"offset": ""},
            pagination_strategy="cursor",
            rate_limit_key="opus",
            item_id_path="items[*].opus_id",
            items_path="items",
        ),
        EndpointSpec(
            name="dynamics",
            callable=_user_method("get_dynamics_new", offset=""),
            credential_required=False,
            params_strategy={"offset": ""},
            pagination_strategy="cursor",
            rate_limit_key="dynamics",
            item_id_path="items[*].id_str",
            items_path="items",
        ),
        EndpointSpec(
            name="audios",
            callable=_user_method("get_audios", pn=1, ps=30, order=user.AudioOrder.PUBDATE),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 30},
            pagination_strategy="page",
            rate_limit_key="audios",
            item_id_path="data[*].id",
            items_path="data",
        ),
        EndpointSpec(
            name="channel_list",
            callable=_user_method("get_channel_list", pn=1, ps=20),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 20},
            pagination_strategy="page",
            rate_limit_key="channel_list",
            item_id_paths=[
                "items_lists.seasons_list[*].meta.season_id",
                "items_lists.series_list[*].meta.series_id",
            ],
            items_path="items_lists",
        ),
        EndpointSpec(
            name="channels",
            callable=fetch_user_channels,
            credential_required=False,
            pagination_strategy="none",
            rate_limit_key="channels",
        ),
        EndpointSpec(
            name="media_list",
            callable=fetch_user_media_list,
            credential_required=False,
            params_strategy={
                "oid": None,
                "ps": 100,
                "direction": True,
                "desc": True,
                "sort_field": int(user.MedialistOrder.PUBDATE.value),
                "tid": 0,
                "with_current": False,
            },
            pagination_strategy="oid",
            rate_limit_key="media_list",
            item_id_paths=["media_list[*].bvid", "list[*].bvid", "items[*].bvid"],
            items_path="media_list",
        ),
    ]


def video_endpoints() -> list[EndpointSpec]:
    return [
        EndpointSpec(
            name="video_detail", callable=fetch_video_detail_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_detail", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_pages", callable=fetch_video_pages_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_pages", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_detail_full", callable=fetch_video_detail_full_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_detail_full", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_ai_conclusion", callable=fetch_video_ai_conclusion_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_ai_conclusion", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_danmaku_snapshot", callable=fetch_video_danmaku_snapshot_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_danmaku_snapshot", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_danmaku_view", callable=fetch_video_danmaku_view_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_danmaku_view", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_danmaku_xml", callable=fetch_video_danmaku_xml_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_danmaku_xml", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_danmakus", callable=fetch_video_danmakus_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_danmakus", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_online", callable=fetch_video_online_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_online", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_pay_coins", callable=fetch_video_pay_coins_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_pay_coins", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_pbp", callable=fetch_video_pbp_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_pbp", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_player_info", callable=fetch_video_player_info_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_player_info", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_private_notes", callable=fetch_video_private_notes_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_private_notes", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_public_notes", callable=fetch_video_public_notes_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_public_notes", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_related", callable=fetch_video_related_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_related", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_relation", callable=fetch_video_relation_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_relation", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_special_dms", callable=fetch_video_special_dms_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_special_dms", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_subtitle", callable=fetch_video_subtitle_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_subtitle", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_up_mid", callable=fetch_video_up_mid_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_up_mid", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_snapshot", callable=fetch_video_snapshot_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_snapshot", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_download_url", callable=fetch_video_download_url_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_download_url", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_is_episode", callable=fetch_video_is_episode_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_is_episode", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_is_forbid_note", callable=fetch_video_is_forbid_note_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="video_is_forbid_note", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
        EndpointSpec(
            name="video_chargers", callable=fetch_video_chargers_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="video_chargers", kind="item",
            source_endpoint="videos", extract_items=_extract_bvids_from_videos,
        ),
    ]


def content_endpoints() -> list[EndpointSpec]:
    return [
        EndpointSpec(
            name="article_detail", callable=fetch_article_detail_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="article_detail", kind="item",
            source_endpoint="articles",
            extract_items=_extract_cvids_from_articles,
            skip_item=_skip_legacy_article_detail_item,
        ),
        EndpointSpec(
            name="opus_detail", callable=fetch_opus_detail_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="opus_detail", kind="item",
            source_endpoint="opus",
            extract_items=_extract_opus_ids_from_opus,
        ),
        EndpointSpec(
            name="article_list_detail", callable=fetch_article_list_detail_item,
            credential_required=False, pagination_strategy="none",
            rate_limit_key="article_list_detail", kind="item",
            source_endpoint="article_list",
            extract_items=_extract_rlids_from_article_list,
        ),
    ]


def channel_and_upower_endpoints() -> list[EndpointSpec]:
    return [
        EndpointSpec(
            name="user_medal", callable=_user_method("get_user_medal"),
            credential_required=True, pagination_strategy="none",
            rate_limit_key="user_medal",
        ),
        EndpointSpec(
            name="live_info", callable=_user_method("get_live_info"),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="live_info",
        ),
        EndpointSpec(
            name="user_relation", callable=_user_method("get_relation"),
            credential_required=True, pagination_strategy="none",
            rate_limit_key="user_relation",
        ),
        EndpointSpec(
            name="reservation", callable=_user_method("get_reservation"),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="reservation",
        ),
        EndpointSpec(
            name="uplikeimg", callable=_user_method("get_uplikeimg"),
            credential_required=True, pagination_strategy="none",
            rate_limit_key="uplikeimg",
        ),
        EndpointSpec(
            name="top_followers", callable=_user_method("top_followers", since=None),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="top_followers",
        ),
        EndpointSpec(
            name="space_notice", callable=_user_method("get_space_notice"),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="space_notice",
        ),
        EndpointSpec(
            name="all_followings", callable=_user_method("get_all_followings"),
            credential_required=True, pagination_strategy="none",
            rate_limit_key="all_followings",
        ),
        EndpointSpec(
            name="followings",
            callable=_user_method(
                "get_followings", pn=1, ps=100, attention=False,
                order=user.OrderType.desc,
            ),
            credential_required=True,
            params_strategy={"pn": 1, "ps": 100},
            pagination_strategy="page", rate_limit_key="followings",
            item_id_path="list[*].mid", items_path="list",
        ),
        EndpointSpec(
            name="followers",
            callable=_user_method("get_followers", pn=1, ps=100, desc=True),
            credential_required=True,
            params_strategy={"pn": 1, "ps": 100},
            pagination_strategy="page", rate_limit_key="followers",
            item_id_path="list[*].mid", items_path="list",
        ),
        EndpointSpec(
            name="same_followers",
            callable=_user_method("get_self_same_followers", pn=1, ps=50),
            credential_required=True,
            params_strategy={"pn": 1, "ps": 50},
            pagination_strategy="page", rate_limit_key="same_followers",
            item_id_path="list[*].mid", items_path="list",
        ),
        EndpointSpec(
            name="top_videos", callable=_user_method("get_top_videos"),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="top_videos",
        ),
        EndpointSpec(
            name="masterpiece",
            callable=lambda uid, cred=None, **kw: (
                _wrap_list_result(
                    user.User(uid, credential=cred).get_masterpiece()
                )
            ),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="masterpiece",
        ),
        EndpointSpec(
            name="article_list",
            callable=_user_method("get_article_list", order=user.ArticleListOrder.LATEST),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="article_list",
        ),
        EndpointSpec(
            name="cheese", callable=_user_method("get_cheese"),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="cheese",
        ),
        EndpointSpec(
            name="elec_monthly", callable=_user_method("get_elec_user_monthly"),
            credential_required=True, pagination_strategy="none",
            rate_limit_key="elec_monthly",
        ),
        EndpointSpec(
            name="user_fav_tag",
            callable=_user_method("get_user_fav_tag", pn=1, ps=20),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 20},
            pagination_strategy="page", rate_limit_key="user_fav_tag",
        ),
        EndpointSpec(
            name="album",
            callable=lambda uid, cred=None, **kw: (
                user.User(uid, credential=cred).get_album(
                    biz=kw.get("biz", user.AlbumType.ALL),
                    page_num=kw.get("pn", 1),
                    page_size=kw.get("ps", 30),
                )
            ),
            credential_required=False,
            params_strategy={"pn": 1, "ps": 30},
            pagination_strategy="page", rate_limit_key="album",
            items_path="biz_list",
        ),
        EndpointSpec(
            name="channel_videos_season",
            callable=lambda sid, cred=None, **kw: (
                _paginate_channel_videos("season", kw["_uid"], int(sid), cred)
            ),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="channel_videos_season", kind="item",
            source_endpoint="channel_list",
            extract_items=_extract_season_ids, needs_parent_uid=True,
        ),
        EndpointSpec(
            name="channel_videos_series",
            callable=lambda sid, cred=None, **kw: (
                _paginate_channel_videos("series", kw["_uid"], int(sid), cred)
            ),
            credential_required=False, pagination_strategy="none",
            rate_limit_key="channel_videos_series", kind="item",
            source_endpoint="channel_list",
            extract_items=_extract_series_ids, needs_parent_uid=True,
        ),
        EndpointSpec(
            name="upower_qa",
            callable=_user_method("get_upower_qa_list", anchor=0),
            credential_required=True,
            params_strategy={"anchor": 0},
            pagination_strategy="anchor", rate_limit_key="upower_qa",
            item_id_path="list[*].qa_id", items_path="list",
        ),
        EndpointSpec(
            name="upower_qa_detail", callable=fetch_upower_qa_detail_item,
            credential_required=True, pagination_strategy="none",
            rate_limit_key="upower_qa_detail", kind="item",
            source_endpoint="upower_qa",
            extract_items=_extract_qa_ids_from_upower_qa,
            needs_parent_uid=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Build catalog
# ---------------------------------------------------------------------------

def _build_endpoints() -> list[EndpointSpec]:
    return [
        *user_endpoints(),
        *video_endpoints(),
        *content_endpoints(),
        *channel_and_upower_endpoints(),
    ]


ENDPOINTS: list[EndpointSpec] = _build_endpoints()
ENDPOINT_BY_NAME: dict[str, EndpointSpec] = {ep.name: ep for ep in ENDPOINTS}


def get_endpoint(name: str) -> EndpointSpec | None:
    return ENDPOINT_BY_NAME.get(name)


def catalog_manifest() -> list[dict[str, Any]]:
    """Return the contract §5.3 describe_catalog manifest — one entry per endpoint."""
    return [
        {
            "name": ep.name,
            "kind": ep.kind,
            "credential_required": ep.credential_required,
            "pagination_strategy": ep.pagination_strategy,
            "params_strategy": ep.params_strategy,
            "source_endpoint": ep.source_endpoint,
            "needs_parent_uid": ep.needs_parent_uid,
        }
        for ep in ENDPOINTS
    ]


__all__ = [
    "ENDPOINTS",
    "ENDPOINT_BY_NAME",
    "EndpointSpec",
    "catalog_manifest",
    "get_endpoint",
]
