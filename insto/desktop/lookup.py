"""On-demand provider reads: one bounded, fail-fast request budget, no writes.

These two operations are the only desktop capabilities that ask the provider
about an account the user just typed, and they are deliberately the narrowest
possible network surface:

- **Nothing local is touched.** The profile's `config.toml` is read for the
  token; no database is opened, no `cli_history` row is written, no snapshot is
  saved and nothing is created under `output/`. A lookup therefore runs happily
  beside a watch daemon owned by the same profile: neither takes a lock.
- **One budget, fail fast.** 60 seconds covers the whole request including the
  client close, the worker is cancelled at the deadline exactly as credential
  validation cancels its own, and the retry ladder is replaced by a single
  quick retry of a transient failure. A rate limit is reported immediately
  rather than slept on: the CLI's five-attempt ladder can block for minutes,
  which a window waiting on one click cannot afford.
- **Bounded output.** Every string and every list below is capped, so a hostile
  or merely verbose account cannot approach the 2 MiB response limit.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol

from insto.backends._retry import with_retry
from insto.desktop.access import _await_worker, _close, access_code, make_backend
from insto.desktop.configuration import parse_profile_config
from insto.desktop.errors import DesktopError
from insto.desktop.history_params import _PK, MAX_TIME
from insto.desktop.profile import Profile
from insto.exceptions import (
    Banned,
    ProfileDeleted,
    ProfileNotFound,
    ProfilePrivate,
)
from insto.models import Post, Quota
from insto.models import Profile as ProfileDTO
from insto.service import analytics
from insto.service.history import _PROFILE_TRACKED_FIELDS

# The budget class of this module: a "network read". One attempt at the whole
# operation, close included, and no second chance at the provider's expense.
NETWORK_READ_SECONDS = 60.0
CLOSE_RESERVE_SECONDS = 2.0
TRANSIENT_RETRY_SECONDS = 0.25

MAX_COUNT = 9007199254740991
_TERM_CHARACTERS = 120
_CODE_CHARACTERS = 64
_TOP_PLACES = 10
_TOP_TERMS = 20
_TOP_POSTS = 5
_TEXT_CHARACTERS: dict[str, int] = {
    "username": 255,
    "full_name": 255,
    "biography": 2048,
    "external_url": 2048,
    "public_email": 320,
    "public_phone": 64,
    "business_category": 255,
}
_BOOLEANS = ("is_verified", "is_business", "is_private")
_COUNTS = ("follower_count", "following_count", "media_count")


class LookupBackend(Protocol):
    """The provider surface a lookup uses; nothing here mutates anything."""

    async def resolve_target(self, username: str) -> str: ...

    async def get_profile(self, pk: str) -> ProfileDTO: ...

    def iter_user_posts(self, pk: str, *, limit: int | None = None) -> AsyncIterator[Post]: ...

    def get_quota(self) -> Quota: ...

    async def aclose(self) -> None: ...


def _fail_fast() -> Any:
    """One quick retry of a transient failure, and never a rate-limit sleep.

    `insto.backends._retry` stays the single place that decides how a backend
    call is retried; this is that policy configured for an interactive budget.
    """
    return with_retry(
        max_attempts=2,
        base_delay=TRANSIENT_RETRY_SECONDS,
        max_delay=TRANSIENT_RETRY_SECONDS,
        retry_rate_limited=False,
    )


def _credentials(profile: Profile) -> tuple[str, str | None]:
    """The profile's own token and proxy; no database and no service are opened."""
    payload = profile.read_config()
    if payload is None:
        raise DesktopError("not_configured")
    config = parse_profile_config(profile, payload)
    token = config.hiker_token
    if not isinstance(token, str):
        raise DesktopError("not_configured")
    return token, config.hiker_proxy


def _failure(exc: BaseException) -> DesktopError:
    """Name the provider's answer without ever forwarding its text."""
    if isinstance(exc, DesktopError):
        return exc
    if isinstance(exc, (ProfileNotFound, ProfileDeleted)):
        return DesktopError("target_not_found")
    if isinstance(exc, (ProfilePrivate, Banned)):
        # `Banned` is this backend's 403: Instagram itself refused the read for
        # this target (login-walled or restricted). On a public lookup surface
        # that is the private-account answer, not a credential problem — 401
        # remains the only "your access is wrong" signal.
        return DesktopError("target_private")
    return DesktopError(access_code(exc))


async def run(profile: Profile, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run one lookup inside the network-read budget, closing the client always."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + NETWORK_READ_SECONDS
    # Reserve a small part of the same budget for closing the HTTP client, as
    # credential validation does; the deadline is the whole request's, not the
    # provider call's, so a slow close cannot extend it.
    request_deadline = deadline - min(CLOSE_RESERVE_SECONDS, NETWORK_READ_SECONDS / 2)
    token, proxy = _credentials(profile)
    backend: LookupBackend | None = None
    failure: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        backend = make_backend(token, proxy=proxy, retry_decorator=_fail_fast())
        result = await _await_worker(
            asyncio.create_task(_read(backend, operation, params)),
            request_deadline,
            cancel_on_interrupt=True,
        )
    except BaseException as exc:
        failure = exc
    finally:
        if backend is not None:
            try:
                await _close(backend, deadline)
            except BaseException as exc:
                if failure is None or isinstance(exc, asyncio.CancelledError):
                    failure = exc
    if isinstance(failure, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        raise failure
    if failure is not None:
        raise _failure(failure) from None
    assert result is not None
    return result


async def _read(backend: LookupBackend, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    if operation == "lookup.profile":
        return await _profile(backend, params["username"])
    return await _activity(backend, params["target_pk"], params["window"])


async def _profile(backend: LookupBackend, username: str) -> dict[str, Any]:
    """Two requests: resolve the username, then read the profile by pk.

    `user_about` is deliberately not called. Every tracked field below comes
    from the profile payload itself, so the third request the CLI's `/info`
    spends would buy this result nothing.
    """
    pk = await backend.resolve_target(username)
    found = await backend.get_profile(pk)
    if found.access == "deleted":
        raise DesktopError("target_not_found")
    values: dict[str, Any] = {
        name: _text(getattr(found, name), _TEXT_CHARACTERS[name]) for name in _TEXT_CHARACTERS
    }
    values.update({name: bool(getattr(found, name)) for name in _BOOLEANS})
    values.update({name: _count(getattr(found, name)) for name in _COUNTS})
    # Exactly the saved-history vocabulary, in its declaration order, with the
    # same value typing `snapshots.read` reports — and without the avatar and
    # banner hashes, which only a stored snapshot can have. A tracked field
    # this provider cannot supply is named in `unknown_fields` rather than
    # invented, so one renderer serves a lookup and a saved snapshot alike.
    fields: dict[str, Any] = {}
    unknown: list[str] = []
    for name in _PROFILE_TRACKED_FIELDS:
        if name in values:
            fields[name] = values[name]
        else:
            unknown.append(name)
    return {
        "kind": "lookup_profile",
        "target_pk": _identity(pk),
        "access": "private" if found.is_private or found.access == "private" else "public",
        "fields": fields,
        "unknown_fields": unknown,
        "quota_remaining": _remaining(backend),
    }


async def _activity(backend: LookupBackend, target_pk: str, window: int) -> dict[str, Any]:
    """One window of recent posts, then every analysis computed from that list.

    The pk comes from `lookup.profile`, so nothing is resolved again: the fetch
    is the posts fetch and its pages, and no per-post request is ever made.
    """
    posts = [post async for post in backend.iter_user_posts(target_pk, limit=window)]
    try:
        geo = analytics.compute_geo_fingerprint(
            posts, target=target_pk, limit=window, top=_TOP_PLACES
        )
        timeline = analytics.compute_timeline(posts, target=target_pk, limit=window)
        hashtags = analytics.extract_hashtags(posts, target=target_pk, limit=window, top=_TOP_TERMS)
        mentions = analytics.extract_mentions(posts, target=target_pk, limit=window, top=_TOP_TERMS)
        locations = analytics.extract_locations(
            posts, target=target_pk, limit=window, top=_TOP_TERMS
        )
        likes = analytics.aggregate_likes(posts, target=target_pk, limit=window, top=_TOP_POSTS)
    except (ValueError, OverflowError, OSError):
        # Only a payload no analysis can represent gets here (an unrepresentable
        # timestamp, say). It is reported, never half-answered.
        raise DesktopError("access_unconfirmed") from None
    return {
        "kind": "lookup_activity",
        "target_pk": target_pk,
        "window": window,
        "analyzed": len(posts),
        "geo": {
            "geotagged": _count(geo.geotagged),
            "anchor": _place(geo.anchor),
            "centroid": _centroid(geo.centroid_lat, geo.centroid_lng),
            "radius_km": _distance(geo.radius_km),
            "places": [
                place for place in map(_place, geo.places[:_TOP_PLACES]) if place is not None
            ],
        },
        "timeline": {
            "hour_of_day": [_count(value) for value in timeline.hour_of_day[:24]],
            "day_of_week": [_count(value) for value in timeline.day_of_week[:7]],
            "first_post_at": _moment(timeline.first_post_ts),
            "last_post_at": _moment(timeline.last_post_ts),
        },
        "hashtags": _terms(hashtags.items),
        "mentions": _terms(mentions.items),
        "locations": _terms(locations.items),
        "likes": {
            "total": _count(likes.total_likes),
            "average": _average(likes.avg_likes),
            "top_posts": [
                {"code": _text(code, _CODE_CHARACTERS) or "", "like_count": _count(value)}
                for code, value in likes.top_posts[:_TOP_POSTS]
            ],
        },
        "quota_remaining": _remaining(backend),
    }


def _identity(pk: str) -> str:
    """The pk the app will hand back to `lookup.activity`, in its own form."""
    if not isinstance(pk, str) or _PK.fullmatch(pk) is None:
        raise DesktopError("access_unconfirmed")
    return pk


def _text(value: Any, characters: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DesktopError("access_unconfirmed")
    return value[:characters]


def _count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DesktopError("access_unconfirmed")
    number: int = value
    return min(max(number, 0), MAX_COUNT)


def _moment(value: int | None) -> int | None:
    if value is None:
        return None
    return value if 0 <= value <= MAX_TIME else None


def _coordinate(value: float | None) -> float | None:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _distance(value: float | None) -> float | None:
    number = _coordinate(value)
    return None if number is None else round(max(number, 0.0), 3)


def _average(value: float) -> float:
    number = _coordinate(value)
    return 0.0 if number is None else round(max(number, 0.0), 3)


def _centroid(lat: float | None, lng: float | None) -> dict[str, float] | None:
    latitude, longitude = _coordinate(lat), _coordinate(lng)
    if latitude is None or longitude is None:
        return None
    return {"lat": latitude, "lng": longitude}


def _place(place: analytics.GeoPlace | None) -> dict[str, Any] | None:
    """A place only exists on the wire with both coordinates JSON can carry."""
    if place is None:
        return None
    coordinates = _centroid(place.lat, place.lng)
    if coordinates is None:
        return None
    return {
        "name": _text(place.name, _TERM_CHARACTERS) or "",
        "lat": coordinates["lat"],
        "lng": coordinates["lng"],
        "count": _count(place.count),
    }


def _terms(items: Iterable[tuple[str, int]]) -> list[dict[str, Any]]:
    return [
        {"key": _text(key, _TERM_CHARACTERS) or "", "count": _count(value)}
        for key, value in list(items)[:_TOP_TERMS]
    ]


def _remaining(backend: LookupBackend) -> int | None:
    """What the response headers of this very call said, or null.

    Never a `/sys/balance` request: a lookup costs what the app promised and
    not one request more.
    """
    remaining = backend.get_quota().remaining
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        return None
    return min(remaining, MAX_COUNT)
