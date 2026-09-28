"""Per-identity profile access for multi-user OAuth deployments (Stampede local patch).

Policy lives in the launch profile's ``config.yaml``::

    dashboard:
      profile_access:
        admins: [patrick@stampede.ai]          # full dashboard + every unowned profile
        shared_profiles: [support, wifi]        # unowned profiles any signed-in person may open

Disabled (every function is a no-op / permissive) when ``admins`` is empty, so an install
without the block behaves exactly like upstream.

Rules for an authenticated identity:
  * a profile whose ``profile.yaml`` carries ``owner_email`` is visible ONLY to that owner;
  * admins see every unowned profile (default included);
  * everyone else sees their own profile plus ``shared_profiles``; the launch/default profile
    and every other unowned profile are invisible, and an omitted profile resolves to THEIR
    OWN home, never the launch profile's.
No identity (loopback / token / stdio callers, which already hold the host's session token)
keeps upstream behaviour for unowned profiles and fails closed on owned ones.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterable, List, Optional

_request_identity: ContextVar[str] = ContextVar("profile_access_identity", default="")
_policy_cache: dict = {"mtime": None, "policy": None}


def normalize_identity(user_id: object) -> str:
    """``stampede:Eamonn@X`` / ``Eamonn@X`` -> ``eamonn@x``."""
    ident = str(user_id or "").strip().lower()
    if ":" in ident and "@" in ident and ident.index(":") < ident.index("@"):
        ident = ident.split(":", 1)[1]
    return ident


def set_request_identity(user_id: object) -> Token:
    return _request_identity.set(normalize_identity(user_id))


def reset_request_identity(token: Token) -> None:
    _request_identity.reset(token)


def current_request_identity() -> str:
    return _request_identity.get()


def _launch_config_path() -> Path:
    from hermes_cli.profiles import get_profile_dir
    return Path(get_profile_dir("default")) / "config.yaml"


def policy() -> dict:
    """``{admins: set, shared: set, enabled: bool}`` re-read when config.yaml changes."""
    path = _launch_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _policy_cache["policy"] is not None and _policy_cache["mtime"] == mtime:
        pol = _policy_cache["policy"]
        return dict(pol, enabled=pol["enabled"] and not _off_switch())
    admins: set = set()
    shared: set = set()
    try:
        import yaml
        data = yaml.safe_load(path.read_text()) or {}
        block = ((data.get("dashboard") or {}).get("profile_access") or {})
        admins = {normalize_identity(e) for e in (block.get("admins") or []) if e}
        shared = {str(p).strip().lower() for p in (block.get("shared_profiles") or []) if p}
    except Exception:
        pass
    pol = {"admins": admins, "shared": shared, "enabled": bool(admins)}
    _policy_cache.update(mtime=mtime, policy=pol)
    return dict(pol, enabled=pol["enabled"] and not _off_switch())


def is_admin(identity: str) -> bool:
    pol = policy()
    return not pol["enabled"] or normalize_identity(identity) in pol["admins"]


def restricted(identity: str) -> bool:
    """True when *identity* is a signed-in non-admin under an enabled policy."""
    ident = normalize_identity(identity)
    return bool(ident) and policy()["enabled"] and ident not in policy()["admins"]


def _owner_of(profile_dir: Path) -> str:
    from hermes_cli.profiles import read_profile_meta
    try:
        meta = read_profile_meta(Path(profile_dir))
    except Exception:
        return ""
    owner = meta.get("owner_email") if isinstance(meta, dict) else ""
    if not owner:
        try:
            import yaml
            owner = (yaml.safe_load((Path(profile_dir) / "profile.yaml").read_text()) or {}).get("owner_email", "")
        except Exception:
            owner = ""
    return normalize_identity(owner)


def _is_launch(profile_dir: Path) -> bool:
    from hermes_cli.profiles import get_profile_dir
    try:
        return Path(profile_dir).resolve() == Path(get_profile_dir("default")).resolve()
    except Exception:
        return False


def profile_visible_to(profile_dir: Path, *, identity: str, name: str = "") -> bool:
    ident = normalize_identity(identity)
    owner = _owner_of(profile_dir)
    if owner:
        return bool(ident) and ident == owner
    if not restricted(ident):
        return True
    if _is_launch(profile_dir):
        return False
    pname = (name or Path(profile_dir).name).strip().lower()
    return pname in policy()["shared"]


def filter_profiles(profiles: Iterable, *, identity: str) -> List:
    """Rows (anything with ``.path``/``.name`` or ``path``/``name`` keys) visible to *identity*."""
    out = []
    for p in profiles:
        path = getattr(p, "path", None) or (p.get("path") if isinstance(p, dict) else None)
        name = getattr(p, "name", None) or (p.get("name") if isinstance(p, dict) else "") or ""
        if path and profile_visible_to(Path(str(path)), identity=identity, name=str(name)):
            out.append(p)
    return out


def own_profile_home(identity: str) -> Optional[Path]:
    """The owner-bound profile home for *identity*, or None."""
    ident = normalize_identity(identity)
    if not ident:
        return None
    from hermes_cli.profiles import _get_profiles_root
    root = Path(_get_profiles_root())
    try:
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    except OSError:
        return None
    for d in dirs:
        if _owner_of(d) == ident:
            return d
    return None


# ---- profile RPC gate (desktop/websocket twin of the REST routes) --------------------------

# ws methods that target a profile by params.name (or create one): non-admins may only
# touch their own visible profiles; everyone else is refused before the handler runs.
_PROFILE_NAMED_METHODS = {
    "profiles.describe", "profiles.configure", "profiles.set_asset", "profiles.get_asset",
    "profiles.remember_onboarding", "profiles.create",
}
# methods only admins may reach at all (launch-profile plumbing over ws)
_PROFILE_ADMIN_ONLY = {"profiles.create"}


def guard_profile_rpc(name: str, params: dict, auth_identity: Optional[dict]) -> None:
    """Raise PermissionError for a profile RPC a signed-in non-admin may not run."""
    ident = normalize_identity((auth_identity or {}).get("user_id"))
    if not restricted(ident):
        return
    if name in _PROFILE_ADMIN_ONLY:
        raise PermissionError("This area is limited to workspace admins.")
    if name in _PROFILE_NAMED_METHODS:
        target = str((params or {}).get("name") or "").strip()
        if not target:
            raise PermissionError("No agent selected for this action.")
        from hermes_cli import profiles as profiles_mod
        try:
            home = Path(profiles_mod.get_profile_dir(target))
        except ValueError:
            home = None
        if home is None or not home.is_dir() or not profile_visible_to(home, identity=ident, name=target):
            raise PermissionError("That agent is not available.")


# ---- dashboard REST gate -------------------------------------------------------------------

async def call_as(request, session, call_next):
    """Serve *request* under *session*'s identity: bind it for route handlers and refuse
    admin-only REST for signed-in non-admins (403, JSON)."""
    identity = getattr(session, "email", "") or getattr(session, "user_id", "") or ""
    token = set_request_identity(identity)
    try:
        if not rest_allowed(request.url.path, request.method, identity):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "This area is limited to workspace admins."}, status_code=403)
        return await call_next(request)
    finally:
        reset_request_identity(token)

# Paths a signed-in non-admin may call. Everything else under /api/ answers 403: the launch
# profile's config, .env, cron, MCP, plugins, filesystem and logs belong to the host's admins.
_OPEN_PREFIXES = (
    "/api/auth", "/api/ws", "/api/display/ws", "/api/status", "/api/audio/",
    "/api/model/options", "/api/model/recommended-default",
    "/api/profiles/sessions", "/api/profiles/projects",
)


def rest_allowed(path: str, method: str, identity: str) -> bool:
    if not restricted(identity):
        return True
    path = "/" + path.lstrip("/")
    if not path.startswith("/api/"):
        return True  # SPA shell, static assets, auth pages
    if any(path == p.rstrip("/") or path.startswith(p) for p in _OPEN_PREFIXES):
        return True
    if path in ("/api/profiles", "/api/profiles/"):
        return method.upper() == "GET"
    if path.startswith("/api/profiles/"):
        name = path[len("/api/profiles/"):].split("/", 1)[0]
        if name in ("active",):
            return method.upper() == "GET"
        from hermes_cli.profiles import get_profile_dir
        try:
            home = Path(get_profile_dir(name))
        except Exception:
            return False
        return home.is_dir() and profile_visible_to(home, identity=identity, name=name)
    return False


def _off_switch() -> bool:
    """Emergency off-switch without a restart: touch <launch home>/profile_access.off."""
    try:
        return (_launch_config_path().parent / "profile_access.off").exists()
    except Exception:
        return False
