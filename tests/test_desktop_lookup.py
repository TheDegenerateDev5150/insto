"""On-demand lookups: bounded, fail-fast provider reads that write nothing."""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from insto.config import Config
from insto.desktop import lookup
from insto.desktop.configuration import config_bytes
from insto.desktop.errors import MESSAGES, DesktopError
from insto.desktop.lookup_params import validate_params
from insto.desktop.profile import Profile
from insto.exceptions import (
    AuthInvalid,
    BackendError,
    Banned,
    ProfileDeleted,
    ProfileNotFound,
    ProfilePrivate,
    QuotaExhausted,
    RateLimited,
    SchemaDrift,
    Transient,
)
from insto.models import Post, Quota
from insto.models import Profile as ProfileDTO

MOMENT = int(datetime(2026, 9, 15, 10, 30, tzinfo=UTC).timestamp())  # a Tuesday
DAY = 86400


def profile_dto(**overrides):
    values = {
        "pk": "17841400000000001",
        "username": "alice",
        "access": "public",
        "full_name": "Alice Example",
        "biography": "bio line",
        "external_url": "https://example.test/alice",
        "is_verified": True,
        "is_business": False,
        "is_private": False,
        "public_email": "alice@example.test",
        "public_phone": None,
        "business_category": None,
        "follower_count": 1200,
        "following_count": 300,
        "media_count": 87,
        "avatar_url": "https://cdn.example.test/a.jpg",
        "avatar_url_hash": "f" * 64,
    }
    values.update(overrides)
    return ProfileDTO(**values)


def make_post(index, *, place=None, lat=None, lng=None, likes=0, tags=(), mentions=(), when=None):
    return Post(
        pk=str(9000 + index),
        code=f"code{index}",
        taken_at=MOMENT + index * DAY if when is None else when,
        media_type="image",
        caption="caption",
        like_count=likes,
        location_name=place,
        location_pk=None if place is None else f"pk-{place}",
        location_lat=lat,
        location_lng=lng,
        hashtags=list(tags),
        mentions=list(mentions),
    )


class FakeBackend:
    """Records every provider call; nothing here touches a real network."""

    def __init__(self, *, posts=(), page=50, quota=None, profile=None, errors=None, hang=False):
        self.posts = list(posts)
        self.page = page
        self.quota = Quota.unknown() if quota is None else Quota.with_remaining(quota)
        self.profile = profile_dto() if profile is None else profile
        self.errors = dict(errors or {})
        self.hang = hang
        self.calls = []
        self.closed = 0

    def _raise(self, slot):
        error = self.errors.get(slot)
        if error is not None:
            raise error

    async def resolve_target(self, username):
        self.calls.append(("resolve", username))
        if self.hang:
            await asyncio.Event().wait()
        self._raise("resolve")
        return self.profile.pk

    async def get_profile(self, pk):
        self.calls.append(("profile", pk))
        self._raise("profile")
        return self.profile

    async def get_user_about(self, pk):  # pragma: no cover - must never be called
        self.calls.append(("about", pk))
        raise AssertionError("lookup.profile must not spend a request on user_about")

    async def iter_user_posts(self, pk, *, limit=None):
        index = 0
        while True:
            self.calls.append(("posts", pk, limit, index))
            if self.hang:
                await asyncio.Event().wait()
            self._raise("posts")
            chunk = self.posts[index : index + self.page]
            for post in chunk:
                yield post
                index += 1
                if limit is not None and index >= limit:
                    return
            if len(chunk) < self.page:
                return

    def get_quota(self):
        return self.quota

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def lookup_profile(tmp_path):
    """A configured profile with no database and no output directory."""
    profile = Profile(tmp_path / "desktop")
    with profile.locked(initialize=True):
        profile.write_config(config_bytes(profile, "offline-desktop-token"))
        profile.write_state(profile.new_state(remaining=8, desired="stopped"))
    return profile


@pytest.fixture
def constructed(monkeypatch):
    """Capture how the backend was built, and hand back the injected fake."""
    record = {}

    def install(backend):
        def construct(token, *, proxy=None, retry_decorator=None):
            record.update(token=token, proxy=proxy, retry_decorator=retry_decorator)
            return backend

        monkeypatch.setattr(lookup, "make_backend", construct)
        return record

    return install


async def call(profile, operation, params):
    return await lookup.run(profile, operation, validate_params(operation, params))


# --------------------------------------------------------------- lookup.profile


async def test_profile_returns_the_tracked_vocabulary_and_costs_two_requests(
    lookup_profile, constructed
):
    from insto.service.history import _PROFILE_TRACKED_FIELDS

    backend = FakeBackend(quota=4211)
    record = constructed(backend)
    result = await call(lookup_profile, "lookup.profile", {"username": "@Alice"})
    assert result == {
        "kind": "lookup_profile",
        "target_pk": "17841400000000001",
        "access": "public",
        "fields": {
            "username": "alice",
            "full_name": "Alice Example",
            "biography": "bio line",
            "external_url": "https://example.test/alice",
            "is_verified": True,
            "is_business": False,
            "is_private": False,
            "follower_count": 1200,
            "following_count": 300,
            "media_count": 87,
            "public_email": "alice@example.test",
            "public_phone": None,
            "business_category": None,
        },
        "unknown_fields": [],
        "quota_remaining": 4211,
    }
    # Exactly the saved-history vocabulary and order, without the media hashes.
    assert list(result["fields"]) == list(_PROFILE_TRACKED_FIELDS)
    assert not {"avatar", "banner"} & set(result["fields"])
    assert backend.calls == [("resolve", "alice"), ("profile", "17841400000000001")]
    assert backend.closed == 1
    assert record["token"] == "offline-desktop-token" and record["proxy"] is None
    assert record["retry_decorator"] is not None


async def test_profile_reports_a_private_account_without_refusing_it(lookup_profile, constructed):
    constructed(FakeBackend(profile=profile_dto(access="private", is_private=True)))
    result = await call(lookup_profile, "lookup.profile", {"username": "alice"})
    assert result["access"] == "private"
    assert result["fields"]["is_private"] is True


async def test_profile_field_the_provider_cannot_supply_is_named_not_invented(
    lookup_profile, constructed, monkeypatch
):
    monkeypatch.setattr(
        lookup, "_PROFILE_TRACKED_FIELDS", (*lookup._PROFILE_TRACKED_FIELDS, "pronouns")
    )
    constructed(FakeBackend())
    result = await call(lookup_profile, "lookup.profile", {"username": "alice"})
    assert result["unknown_fields"] == ["pronouns"]
    assert "pronouns" not in result["fields"]


async def test_profile_strings_are_bounded(lookup_profile, constructed):
    constructed(FakeBackend(profile=profile_dto(biography="b" * 5000, full_name="f" * 900)))
    fields = (await call(lookup_profile, "lookup.profile", {"username": "alice"}))["fields"]
    assert fields["biography"] == "b" * 2048
    assert fields["full_name"] == "f" * 255


@pytest.mark.parametrize(
    "backend",
    [
        FakeBackend(profile=profile_dto(access="deleted")),
        FakeBackend(errors={"resolve": ProfileNotFound("alice")}),
        FakeBackend(errors={"profile": ProfileDeleted("alice")}),
    ],
    ids=["deleted", "unresolvable", "vanished"],
)
async def test_profile_missing_account_is_target_not_found(lookup_profile, constructed, backend):
    constructed(backend)
    with pytest.raises(DesktopError, match="target_not_found"):
        await call(lookup_profile, "lookup.profile", {"username": "alice"})
    assert backend.closed == 1


async def test_profile_identity_the_app_cannot_reuse_is_refused(lookup_profile, constructed):
    constructed(FakeBackend(profile=profile_dto(pk="017")))
    with pytest.raises(DesktopError, match="access_unconfirmed"):
        await call(lookup_profile, "lookup.profile", {"username": "alice"})


# -------------------------------------------------------------- lookup.activity


def geotagged_window():
    return [
        make_post(0, place="Cafe Zero", lat=52.37, lng=4.89, likes=10, tags=["ams", "coffee"]),
        make_post(1, place="Cafe Zero", lat=52.37, lng=4.89, likes=30, tags=["ams"]),
        make_post(2, place="Museum", lat=52.36, lng=4.88, likes=20, mentions=["@bob"]),
        make_post(3, likes=40, tags=["coffee"], mentions=["Bob"]),
    ]


async def test_activity_analyses_one_window_of_posts(lookup_profile, constructed):
    backend = FakeBackend(posts=geotagged_window(), quota=4100)
    constructed(backend)
    result = await call(
        lookup_profile, "lookup.activity", {"target_pk": "17841400000000001", "window": 50}
    )
    assert result["kind"] == "lookup_activity"
    assert result["target_pk"] == "17841400000000001"
    assert result["window"] == 50 and result["analyzed"] == 4
    geo = result["geo"]
    assert geo["geotagged"] == 3
    assert geo["anchor"] == {"name": "Cafe Zero", "lat": 52.37, "lng": 4.89, "count": 2}
    assert geo["centroid"]["lat"] == pytest.approx(52.3666, abs=1e-3)
    assert 0.0 <= geo["radius_km"] < 5.0
    assert [place["name"] for place in geo["places"]] == ["Cafe Zero", "Museum"]
    timeline = result["timeline"]
    assert sum(timeline["hour_of_day"]) == 4 and timeline["hour_of_day"][10] == 4
    assert len(timeline["hour_of_day"]) == 24 and len(timeline["day_of_week"]) == 7
    assert timeline["day_of_week"][1] == 1  # the window starts on a Tuesday
    assert timeline["first_post_at"] == MOMENT
    assert timeline["last_post_at"] == MOMENT + 3 * DAY
    assert result["hashtags"] == [{"key": "ams", "count": 2}, {"key": "coffee", "count": 2}]
    assert result["mentions"] == [{"key": "bob", "count": 2}]
    assert result["locations"] == [{"key": "Cafe Zero", "count": 2}, {"key": "Museum", "count": 1}]
    assert result["likes"]["total"] == 100
    assert result["likes"]["average"] == 25.0
    assert result["likes"]["top_posts"][0] == {"code": "code3", "like_count": 40}
    assert len(result["likes"]["top_posts"]) == 4
    assert result["quota_remaining"] == 4100
    # One posts fetch, no per-post request, and the username is not resolved again.
    assert backend.calls == [("posts", "17841400000000001", 50, 0)]
    assert backend.closed == 1


async def test_activity_of_an_account_without_posts_is_the_same_shape(lookup_profile, constructed):
    constructed(FakeBackend(posts=[]))
    result = await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": 12})
    assert result == {
        "kind": "lookup_activity",
        "target_pk": "7",
        "window": 12,
        "analyzed": 0,
        "geo": {
            "geotagged": 0,
            "anchor": None,
            "centroid": None,
            "radius_km": None,
            "places": [],
        },
        "timeline": {
            "hour_of_day": [0] * 24,
            "day_of_week": [0] * 7,
            "first_post_at": None,
            "last_post_at": None,
        },
        "hashtags": [],
        "mentions": [],
        "locations": [],
        "likes": {"total": 0, "average": 0.0, "top_posts": []},
        "quota_remaining": None,
    }


@pytest.mark.parametrize(
    "window,page,fetches,analyzed", [(12, 50, 1, 12), (30, 12, 3, 30), (50, 50, 1, 50)]
)
async def test_activity_stops_at_the_window(
    lookup_profile, constructed, window, page, fetches, analyzed
):
    backend = FakeBackend(posts=[make_post(i) for i in range(60)], page=page)
    constructed(backend)
    result = await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": window})
    assert result["analyzed"] == analyzed == result["window"]
    assert len(backend.calls) == fetches
    assert all(call[0] == "posts" and call[2] == window for call in backend.calls)


async def test_activity_bounds_every_list_and_string(lookup_profile, constructed):
    posts = [
        make_post(
            index,
            place=f"{index:03d}-" + "P" * 300,
            lat=1.0 + index / 1000,
            lng=2.0,
            likes=index,
            tags=[f"tag{index}", "x" * 300],
            mentions=[f"user{index}"],
        )
        for index in range(50)
    ]
    constructed(FakeBackend(posts=posts))
    result = await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": 50})
    assert len(result["hashtags"]) == 20
    assert len(result["mentions"]) == 20
    assert len(result["locations"]) == 20
    assert len(result["geo"]["places"]) == 10
    assert len(result["likes"]["top_posts"]) == 5
    assert all(len(item["key"]) <= 120 for item in result["hashtags"] + result["locations"])
    assert len(result["geo"]["anchor"]["name"]) == 120
    assert len({item["key"] for item in result["locations"]}) == 20
    assert all(len(item["code"]) <= 64 for item in result["likes"]["top_posts"])
    encoded = json.dumps(result, ensure_ascii=True, allow_nan=False).encode("ascii")
    assert len(encoded) < 64 * 1024


async def test_activity_of_a_private_or_restricted_account(lookup_profile, constructed):
    for error in (ProfilePrivate("alice"), Banned("login-walled")):
        backend = FakeBackend(errors={"posts": error})
        constructed(backend)
        with pytest.raises(DesktopError, match="target_private"):
            await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": 12})
        assert backend.closed == 1


async def test_activity_of_a_vanished_account(lookup_profile, constructed):
    constructed(FakeBackend(errors={"posts": ProfileNotFound("7")}))
    with pytest.raises(DesktopError, match="target_not_found"):
        await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": 12})


# ------------------------------------------------------------- budget and errors


@pytest.mark.parametrize(
    "error,code",
    [
        (AuthInvalid("token-secret"), "invalid_token"),
        (QuotaExhausted("balance"), "quota_exhausted"),
        (RateLimited(300.0, "cooldown"), "rate_limited"),
        (Transient("blip"), "network_error"),
        (SchemaDrift("user", "pk"), "access_unconfirmed"),
        (BackendError("surprise"), "access_unconfirmed"),
        (RuntimeError("bug"), "access_unconfirmed"),
        (TimeoutError("slow"), "operation_timeout"),
    ],
)
async def test_provider_failures_map_to_static_codes(lookup_profile, constructed, error, code):
    backend = FakeBackend(errors={"resolve": error})
    constructed(backend)
    with pytest.raises(DesktopError) as raised:
        await call(lookup_profile, "lookup.profile", {"username": "alice"})
    assert raised.value.code == code
    assert str(error) not in str(raised.value)
    assert code in MESSAGES
    assert backend.closed == 1


async def test_rate_limit_is_reported_at_once_and_a_blip_is_retried_once():
    loop = asyncio.get_running_loop()
    attempts = []

    @lookup._fail_fast()
    async def rate_limited():
        attempts.append(True)
        raise RateLimited(retry_after=300.0)

    started = loop.time()
    with pytest.raises(RateLimited):
        await rate_limited()
    # The CLI ladder would have slept for the announced cooldown here.
    assert attempts == [True] and loop.time() - started < 1.0

    transient = []

    @lookup._fail_fast()
    async def once():
        transient.append(True)
        if len(transient) == 1:
            raise Transient("blip")
        return "ok"

    started = loop.time()
    assert await once() == "ok"
    assert len(transient) == 2
    assert loop.time() - started <= lookup.TRANSIENT_RETRY_SECONDS + 1.0

    @lookup._fail_fast()
    async def broken():
        transient.append(True)
        raise Transient("blip")

    with pytest.raises(Transient):
        await broken()
    assert len(transient) == 4  # one retry, never a ladder


async def test_hanging_provider_is_cancelled_within_the_budget(
    lookup_profile, constructed, monkeypatch
):
    monkeypatch.setattr(lookup, "NETWORK_READ_SECONDS", 0.05)
    backend = FakeBackend(hang=True)
    constructed(backend)
    started = asyncio.get_running_loop().time()
    with pytest.raises(DesktopError, match="operation_timeout"):
        await call(lookup_profile, "lookup.profile", {"username": "alice"})
    assert asyncio.get_running_loop().time() - started < 5.0
    assert backend.closed == 1


async def test_unconfigured_profile_never_constructs_a_provider(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(lookup, "make_backend", lambda *a, **k: called.append(a))
    profile = Profile(tmp_path / "empty")
    with pytest.raises(DesktopError, match="not_configured"):
        await call(profile, "lookup.profile", {"username": "alice"})
    assert called == []


async def test_the_profiles_own_proxy_is_used(lookup_profile, constructed, monkeypatch):
    payload = lookup_profile.read_config()
    config = lookup.parse_profile_config(lookup_profile, payload)
    monkeypatch.setattr(
        lookup,
        "parse_profile_config",
        lambda profile, raw: Config(
            **{
                **{
                    field: getattr(config, field)
                    for field in config.__dataclass_fields__
                    if field != "hiker_proxy"
                },
                "hiker_proxy": "socks5://127.0.0.1:9050",
            }
        ),
    )
    record = constructed(FakeBackend())
    await call(lookup_profile, "lookup.profile", {"username": "alice"})
    assert record["proxy"] == "socks5://127.0.0.1:9050"


# ------------------------------------------------------------------------ purity


def inventory(root):
    return {
        str(path.relative_to(root)): path.stat().st_mtime_ns
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


async def test_a_lookup_writes_nothing_at_all(lookup_profile, constructed):
    before = inventory(lookup_profile.root)
    constructed(FakeBackend(posts=geotagged_window(), quota=12))
    await call(lookup_profile, "lookup.profile", {"username": "alice"})
    await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": 12})
    assert inventory(lookup_profile.root) == before
    assert not (lookup_profile.home / "store.db").exists()
    assert not (lookup_profile.home / "output").exists()
    assert not (lookup_profile.home / "cli_history").exists()
    assert lookup_profile.read_state()["quota_remaining"] == 8


async def test_a_lookup_opens_no_database_and_takes_no_lock(
    lookup_profile, constructed, monkeypatch
):
    import sqlite3

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: pytest.fail("a lookup must not open a database")
    )
    monkeypatch.setattr(
        Profile, "locked", lambda *a, **k: pytest.fail("a lookup must not lock the profile")
    )
    constructed(FakeBackend(posts=geotagged_window()))
    await call(lookup_profile, "lookup.activity", {"target_pk": "7", "window": 12})


async def test_the_process_path_answers_a_hanging_provider_with_operation_timeout(
    lookup_profile, tmp_path
):
    """The whole child process: a cancelled worker becomes one static envelope."""
    import os
    import subprocess
    import sys

    script = """
import asyncio
import sys

import insto.desktop.lookup as lookup


class Hanging:
    async def resolve_target(self, username):
        await asyncio.Event().wait()

    def get_quota(self):
        raise AssertionError("never reached")

    async def aclose(self):
        return None


lookup.NETWORK_READ_SECONDS = 0.2
lookup.make_backend = lambda *args, **kwargs: Hanging()

from insto.desktop.dispatch import handle

sys.stdout.buffer.write(asyncio.run(handle(sys.stdin.buffer.read())))
"""
    request = (
        json.dumps(
            {
                "protocol_version": 1,
                "request_id": "hang",
                "operation": "lookup.profile",
                "params": {"username": "alice"},
            }
        )
        + "\n"
    ).encode()
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        input=request,
        capture_output=True,
        cwd=tmp_path,
        env={**os.environ, "INSTO_DESKTOP_ROOT": str(lookup_profile.root)},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    response = json.loads(result.stdout)
    assert response["request_id"] == "hang"
    assert response["error"] == {
        "code": "operation_timeout",
        "message": MESSAGES["operation_timeout"][0],
        "retryable": False,
    }
    assert not (lookup_profile.home / "store.db").exists()
