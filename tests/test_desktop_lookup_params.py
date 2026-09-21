"""Lookup parameters are validated before any profile or provider is loaded."""

import importlib
import sys

import pytest

from insto.desktop.errors import DesktopError
from insto.desktop.lookup_params import CAPABILITIES, WINDOWS, validate_params


def test_capabilities_and_normalized_results():
    assert CAPABILITIES == ("lookup.profile", "lookup.activity")
    assert WINDOWS == (12, 30, 50)
    assert validate_params("lookup.profile", {"username": "@Alice"}) == {"username": "alice"}
    assert validate_params("lookup.activity", {"target_pk": "1" * 64, "window": 12}) == {
        "target_pk": "1" * 64,
        "window": 12,
    }


def test_username_uses_exactly_the_watches_add_rule():
    from insto.service.history import _canonical_watch_user

    for raw in ("@@Alice ", "@alice", "ALICE", "  alice  "):
        assert validate_params("lookup.profile", {"username": raw})["username"] == "alice"
        assert _canonical_watch_user(raw) == "alice"
    assert validate_params("lookup.profile", {"username": "A" * 255})["username"] == "a" * 255


@pytest.mark.parametrize(
    "username",
    [" @alice", "alice bob", "ali/ce", ".", "..", "", "a" * 256, "юзер", 5, None, True],
)
def test_invalid_usernames(username):
    with pytest.raises(DesktopError, match="invalid_params"):
        validate_params("lookup.profile", {"username": username})


@pytest.mark.parametrize(
    "target_pk", ["0", "01", "", "1" * 65, "12a", " 12", "12 ", 12, None, True, "-1"]
)
def test_invalid_target_pk(target_pk):
    with pytest.raises(DesktopError, match="invalid_params"):
        validate_params("lookup.activity", {"target_pk": target_pk, "window": 50})


@pytest.mark.parametrize("window", [True, 12.0, 0, 11, 13, 51, "50", None, -12])
def test_window_is_one_of_the_offered_integers(window):
    with pytest.raises(DesktopError, match="invalid_params"):
        validate_params("lookup.activity", {"target_pk": "7", "window": window})


@pytest.mark.parametrize(
    "operation,params",
    [
        ("lookup.profile", {}),
        ("lookup.profile", {"username": "alice", "window": 12}),
        ("lookup.profile", {"user": "alice"}),
        ("lookup.profile", {"target_pk": "7"}),
        ("lookup.activity", {}),
        ("lookup.activity", {"target_pk": "7"}),
        ("lookup.activity", {"window": 12}),
        ("lookup.activity", {"target_pk": "7", "window": 12, "limit": 1}),
        ("lookup.activity", {"username": "alice", "window": 12}),
        ("watches.add", {"user": "alice"}),
    ],
)
def test_exact_parameter_sets(operation, params):
    with pytest.raises(DesktopError, match="invalid_params"):
        validate_params(operation, params)


@pytest.mark.parametrize("params", [[], None, "x", 5])
def test_params_must_be_an_object(params):
    with pytest.raises(DesktopError, match="invalid_params"):
        validate_params("lookup.profile", params)


def test_parameter_module_has_no_profile_or_provider_imports(monkeypatch):
    for name in ("insto.desktop.profile", "insto.desktop.lookup", "hikerapi", "aiograpi"):
        monkeypatch.setitem(sys.modules, name, None)
    module = importlib.reload(sys.modules["insto.desktop.lookup_params"])
    assert len(module.CAPABILITIES) == len(CAPABILITIES) == 2
    assert module.validate_params("lookup.profile", {"username": "alice"}) == {"username": "alice"}
