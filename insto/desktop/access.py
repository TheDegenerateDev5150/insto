"""Explicit, bounded provider validation without ambient configuration."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from insto._redact import register_secret
from insto.desktop.errors import DesktopError
from insto.exceptions import AuthInvalid, QuotaExhausted, RateLimited, Transient
from insto.models import Quota

if TYPE_CHECKING:
    from insto.backends.hiker import HikerBackend

VALIDATION_SECONDS = 30.0
_PENDING_WORKERS: set[asyncio.Task[Any]] = set()
_T = TypeVar("_T")
RetryDecorator = Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]


class ClosableBackend(Protocol):
    """Every desktop provider client is closed inside its own budget."""

    async def aclose(self) -> None: ...


class AccessBackend(ClosableBackend, Protocol):
    async def validate_access(self) -> Quota: ...


def validate_token(token: str) -> None:
    if (
        not isinstance(token, str)
        or not 4 <= len(token) <= 4096
        or any(not 33 <= ord(character) <= 126 for character in token)
    ):
        raise DesktopError("invalid_params")


def make_backend(
    token: str,
    *,
    proxy: str | None = None,
    retry_decorator: RetryDecorator | None = None,
) -> HikerBackend:
    """Adapt the pinned SDK constructor without mutating process environment.

    SDK BaseAsyncClient currently reads proxy/CA settings and HIKERAPI_HOST.
    Use its BaseClient initializer with an explicit packaged host, then supply
    the same HTTP transport with environment lookup disabled. This deliberately
    narrow adapter is covered by an actual SDK construction regression.

    `proxy` is the profile's own configured proxy (an adopted CLI home may
    carry one); it is validated by the backend's own rule before the SDK is
    built. `retry_decorator` replaces the default five-attempt ladder for
    callers whose budget cannot absorb it. The returned object is the full
    `HikerBackend`; `AccessBackend` is the part credential validation uses.
    """
    import hikerapi
    import httpx
    from hikerapi.__version__ import __host__
    from hikerapi.base import BaseClient

    from insto.backends.hiker import HikerBackend, _validate_proxy_url

    if proxy is not None:
        _validate_proxy_url(proxy)

    class DesktopClient(hikerapi.AsyncClient):  # type: ignore[misc]
        def __init__(self) -> None:
            BaseClient.__init__(self, token=token, timeout=10.0, host=__host__)
            self._client = httpx.AsyncClient(
                base_url=self._url,
                headers=self._headers,
                timeout=10.0,
                trust_env=False,
                follow_redirects=False,
                proxy=proxy,
            )

    return HikerBackend(client=DesktopClient(), retry_decorator=retry_decorator)


async def _await_worker(
    worker: asyncio.Task[_T], deadline: float, *, cancel_on_interrupt: bool = False
) -> _T:
    cancellation: asyncio.CancelledError | None = None
    while not worker.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            # Cancellation cleanup cannot extend credential validation. Retain
            # the cancelled worker until it finishes and consume its exception;
            # the one-shot process owner's deadline is the final hard bound if
            # a third-party close ignores cancellation. No config is committed.
            _PENDING_WORKERS.add(worker)

            def finished(task: asyncio.Task[_T]) -> None:
                _PENDING_WORKERS.discard(task)
                with contextlib.suppress(BaseException):
                    task.result()

            worker.add_done_callback(finished)
            worker.cancel()
            if cancellation is not None:
                raise cancellation
            raise DesktopError("operation_timeout")
        try:
            await asyncio.wait({worker}, timeout=remaining)
        except asyncio.CancelledError as exc:
            if cancellation is None and cancel_on_interrupt:
                worker.cancel()
            cancellation = exc
    if cancellation is not None:
        with contextlib.suppress(BaseException):
            worker.result()
        raise cancellation
    return worker.result()


def access_code(failure: BaseException) -> str:
    """The shared provider-failure mapping for every desktop network call.

    One ladder so credential validation and the lookup reads name the same
    condition with the same code; callers that know more about their own
    surface (a missing or private target) map that before asking here.
    """
    return (
        "invalid_token"
        if isinstance(failure, AuthInvalid)
        else "quota_exhausted"
        if isinstance(failure, QuotaExhausted)
        else "rate_limited"
        if isinstance(failure, RateLimited)
        else "network_error"
        if isinstance(failure, Transient)
        else "operation_timeout"
        if isinstance(failure, TimeoutError)
        else "access_unconfirmed"
    )


async def _close(backend: ClosableBackend, deadline: float) -> None:
    await _await_worker(asyncio.create_task(backend.aclose()), deadline)


async def validate_candidate(token: str) -> int:
    validate_token(token)
    register_secret(token)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + VALIDATION_SECONDS
    backend: AccessBackend | None = None
    failure: BaseException | None = None
    remaining: int | None = None
    try:
        # Reserve a small part of the same budget for closing the HTTP client.
        request_deadline = deadline - min(2.0, VALIDATION_SECONDS / 2)
        backend = make_backend(token)
        quota = await _await_worker(
            asyncio.create_task(backend.validate_access()),
            request_deadline,
            cancel_on_interrupt=True,
        )
        remaining = quota.remaining
        if type(remaining) is not int or remaining < 0:
            raise DesktopError("access_unconfirmed")
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
        if isinstance(failure, DesktopError):
            raise failure from None
        raise DesktopError(access_code(failure)) from None
    assert remaining is not None
    return remaining
