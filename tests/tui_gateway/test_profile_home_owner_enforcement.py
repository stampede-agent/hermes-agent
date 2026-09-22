"""E2E: owner-bound profile enforcement in the bound tui_gateway server.

``tui_gateway.server._profile_home`` is the write-path gate: it resolves a named
profile's HERMES_HOME for session.create / prompt / resume. An owner-bound
profile (``owner_email`` in profile.yaml) must resolve ONLY for the connection's
server-minted ``auth_identity``; a foreign or absent identity must get ``None``
so the call falls back to the launch profile instead of touching another user's
per-user store. Drives the real bound server function with a fake transport.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tui_gateway import server as srv
from tui_gateway.transport import bind_transport, reset_transport


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    """A launch HERMES_HOME plus a sibling profiles/ tree (home-anchored)."""
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # server.py caches its launch home at import; point it at the temp launch dir
    # so a temp profile never short-circuits as "already the launch profile".
    monkeypatch.setattr(srv, "_hermes_home", str(home))
    monkeypatch.setattr(srv, "_served_profile_homes", set())
    return home


def _owned_profile(home, name, owner_email):
    d = home / "profiles" / name
    d.mkdir(parents=True)
    (d / "profile.yaml").write_text(f"owner_email: {owner_email}\n")
    return d


def _unowned_profile(home, name):
    d = home / "profiles" / name
    d.mkdir(parents=True)
    (d / "profile.yaml").write_text("description: shared role profile\n")
    return d


def _transport_with_identity(user_id):
    t = MagicMock()
    t.auth_identity = {"user_id": user_id, "provider": "stampede"}
    return t


def test_owner_resolves_own_profile_foreign_gets_none(launch_home):
    _owned_profile(launch_home, "alice", "alice@stampede.ai")
    _owned_profile(launch_home, "bob", "bob@stampede.ai")

    # Alice's connection resolves her own profile home…
    tok = bind_transport(_transport_with_identity("alice@stampede.ai"))
    try:
        assert srv._profile_home("alice") is not None
        # …but NOT Bob's — foreign owned profile → None (fall back to launch).
        assert srv._profile_home("bob") is None
    finally:
        reset_transport(tok)

    # Bob's connection sees it the other way round.
    tok = bind_transport(_transport_with_identity("bob@stampede.ai"))
    try:
        assert srv._profile_home("bob") is not None
        assert srv._profile_home("alice") is None
    finally:
        reset_transport(tok)


def test_no_identity_cannot_resolve_owned_profile(launch_home):
    """Legacy token / stdio transport carries no auth_identity → owned profile is
    off-limits (fail closed), so a credential without an OAuth identity cannot
    open a per-user store."""
    _owned_profile(launch_home, "carol", "carol@stampede.ai")
    t = MagicMock()
    t.auth_identity = None
    tok = bind_transport(t)
    try:
        assert srv._profile_home("carol") is None
    finally:
        reset_transport(tok)


def test_unowned_profile_resolves_regardless_of_identity(launch_home):
    """Shared role profiles (no owner_email) stay reachable for everyone — the
    gate isolates per-user profiles, it does not partition the shared fleet."""
    _unowned_profile(launch_home, "support")
    for ident in ("alice@stampede.ai", ""):
        tok = bind_transport(_transport_with_identity(ident) if ident else MagicMock(auth_identity=None))
        try:
            assert srv._profile_home("support") is not None
        finally:
            reset_transport(tok)
