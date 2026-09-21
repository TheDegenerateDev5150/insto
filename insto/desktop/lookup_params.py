"""Pure validation for the on-demand lookup reads, before any provider import."""

from __future__ import annotations

from typing import Any

from insto.desktop.errors import DesktopError

# The username and pk rules are not restated here: one rule per field across
# the whole bridge. `watches.add` owns the username form (leading "@", then
# whitespace, then lowercase) and the history operations own the pk form, and
# both raise `invalid_params` on violation, which is what this module promises.
from insto.desktop.history_params import _decimal as _target_pk
from insto.desktop.watch_params import _user as _username

CAPABILITIES = ("lookup.profile", "lookup.activity")
WINDOWS = (12, 30, 50)


def validate_params(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    if type(params) is not dict or operation not in CAPABILITIES:
        raise DesktopError("invalid_params")
    if operation == "lookup.profile":
        if params.keys() != {"username"}:
            raise DesktopError("invalid_params")
        return {"username": _username(params["username"])}
    if params.keys() != {"target_pk", "window"}:
        raise DesktopError("invalid_params")
    window = params["window"]
    # An actual integer from the fixed offered set: the cost the app states
    # before the click is only honest for a window the core will really use.
    if type(window) is not int or window not in WINDOWS:
        raise DesktopError("invalid_params")
    return {"target_pk": _target_pk(params["target_pk"]), "window": window}
