"""Per-request authenticated identity for dashboard REST routes.

The gated auth middleware (:mod:`hermes_cli.dashboard_auth.middleware`) resolves a
provider-verified :class:`Session` and publishes ``user_id`` here so sync route
handlers (including those run in the AnyIO threadpool — contextvars are copied
into ``run_sync`` workers) can scope multi-user surfaces like the profile rail
without threading a ``Request`` parameter through every signature (which would
also poison the sidebar singleflight cache key).

Empty identity = no authenticated user (ungated/loopback/token-auth requests):
owner-bound surfaces fail closed on it, shared surfaces behave as before.
"""
from __future__ import annotations

from contextvars import ContextVar, Token

_request_identity: ContextVar[str] = ContextVar("dashboard_auth_identity", default="")


def set_request_identity(user_id: str) -> Token:
    return _request_identity.set((user_id or "").strip().lower())


def reset_request_identity(token: Token) -> None:
    _request_identity.reset(token)


def current_request_identity() -> str:
    return _request_identity.get()
