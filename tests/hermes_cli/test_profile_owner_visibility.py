"""Owner-bound profile visibility (multi-user OAuth auto-provisioning).

A profile with ``owner_email`` in profile.yaml must be visible ONLY to that
identity; unowned profiles stay visible to everyone (no regression for the
shared fleet / token / loopback callers). These are behaviour contracts on
the gate primitives the WS ``profiles.list``, REST ``/api/profiles`` fan-out,
and ``tui_gateway._profile_home`` enforcement all share.
"""
import pytest
from pathlib import Path


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _make_profile(home, name, owner_email=None):
    d = home / "profiles" / name
    d.mkdir(parents=True)
    meta = {"description": "test"}
    if owner_email:
        meta["owner_email"] = owner_email
    (d / "profile.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")
    return d


def test_owner_bound_profile_only_visible_to_owner(profile_env):
    from hermes_cli import profiles as pm
    owned = _make_profile(profile_env, "alice", owner_email="alice@stampede.ai")
    assert pm.profile_visible_to(owned, identity="alice@stampede.ai") is True
    assert pm.profile_visible_to(owned, identity="Alice@Stampede.ai") is True  # case-insensitive
    assert pm.profile_visible_to(owned, identity="alice") is True  # local-part accepted
    assert pm.profile_visible_to(owned, identity="bob@stampede.ai") is False
    assert pm.profile_visible_to(owned, identity="") is False  # no identity → fail closed


def test_unowned_profile_visible_to_everyone(profile_env):
    from hermes_cli import profiles as pm
    shared = _make_profile(profile_env, "support")  # no owner_email
    assert pm.profile_visible_to(shared, identity="bob@stampede.ai") is True
    assert pm.profile_visible_to(shared, identity="") is True


def test_owner_email_roundtrips_through_profile_meta(profile_env):
    from hermes_cli import profiles as pm
    d = _make_profile(profile_env, "carol")
    pm.write_profile_meta(d, owner_email="Carol@Stampede.ai")
    assert pm.read_profile_meta(d)["owner_email"] == "carol@stampede.ai"  # normalised lower
    pm.write_profile_meta(d, owner_email="")  # clear
    assert pm.read_profile_meta(d)["owner_email"] == ""
    assert "owner_email" not in (d / "profile.yaml").read_text()
