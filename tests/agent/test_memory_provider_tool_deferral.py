"""Memory-provider tool deferral behind the tool_search bridge (kanban t_eb99c448).

Invariants:
- With ``memory.provider_tool_deferral.enabled`` AND the bridge on the surface,
  injection attaches ONLY the eager provider tools; every provider name stays a
  valid tool name (direct calls and skills gating keep working).
- Policy off (default) or no bridge on the surface → classic full-eager injection.
- The between-turns rebuild (``refresh_agent_mcp_tools`` → ``_reinject_post_build_tools``)
  mirrors the deferral: only the eager pair re-attaches to the wire, deferred verbs
  stay valid tool names. An unconditional re-append there silently re-fattened every
  live turn (the 91-tool / 41,633-token Fleet regression).
- Deferred provider tools classify as deferrable (so the bridge catalog, search
  and tool_call validation accept them); eager ones do not.
- A bridge ``tool_call`` targeting a deferred provider tool executes through the
  session-bound MemoryManager — never the global registry (providers are
  per-session objects; a global dispatcher would write into the wrong session).
- skills index_style: "names" demotes every category, "trigger" caps
  descriptions, "full" is unchanged.
"""

import json
import time
from types import SimpleNamespace

import pytest

from agent import memory_manager as mm
from agent.memory_provider import MemoryProvider

EAGER = ["mem_recall", "mem_remember"]
ALL_TOOLS = ["mem_recall", "mem_remember", "mem_sleep", "mem_export"]


def _schema(name: str, desc: str = "desc") -> dict:
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {}}}


class _Provider(MemoryProvider):
    """Minimal provider exposing four tools and echoing calls."""

    def __init__(self):
        self.calls = []

    @property
    def name(self) -> str:
        return "memtest"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        return None

    def get_tool_schemas(self):
        return [_schema(n) for n in ALL_TOOLS]

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, dict(args or {})))
        return json.dumps({"handled": tool_name})


@pytest.fixture()
def provider_env(monkeypatch):
    """Fresh store/caches + a MemoryManager with the fake provider registered."""
    monkeypatch.setattr(mm, "_PROVIDER_TOOL_DEFS", {})
    monkeypatch.setattr(mm, "_PROVIDER_DEFS_GENERATION", 0)
    monkeypatch.setattr(mm, "_policy_cache", {})
    monkeypatch.setattr(mm, "_deferred_names_cache", {})
    provider = _Provider()
    manager = mm.MemoryManager()
    manager.add_provider(provider)
    return provider, manager


def _force_policy(monkeypatch, enabled: bool, scope: str = "test-scope"):
    """Pin the policy for a fixed scope key (the caches are per-config-path now)."""
    monkeypatch.setattr(mm, "_policy_scope_key", lambda: scope)
    monkeypatch.setattr(mm, "_policy_cache",
                        {scope: (time.time() + 3600,
                                 {"enabled": enabled, "eager_tools": list(EAGER)})})
    monkeypatch.setattr(mm, "_deferred_names_cache", {})


def _agent(manager, *, bridge: bool):
    tools = [{"type": "function", "function": {"name": "memory"}}]
    if bridge:
        tools.append({"type": "function", "function": {"name": "tool_search"}})
    return SimpleNamespace(
        _memory_manager=manager, tools=tools,
        valid_tool_names={t["function"]["name"] for t in tools},
        enabled_toolsets=None, disabled_toolsets=None,
    )


def test_injection_defers_cold_tail_when_policy_on(provider_env, monkeypatch):
    _provider, manager = provider_env
    _force_policy(monkeypatch, True)
    agent = _agent(manager, bridge=True)

    added = mm.inject_memory_provider_tools(agent)

    attached = {t["function"]["name"] for t in agent.tools if isinstance(t, dict)}
    assert added == 2
    assert attached == {"memory", "tool_search", *EAGER}
    # every provider name stays valid (direct dispatch + skills gating)
    assert set(ALL_TOOLS) <= agent.valid_tool_names
    assert mm.deferred_provider_tool_names() == {"mem_sleep", "mem_export"}


def test_injection_full_eager_when_policy_off(provider_env, monkeypatch):
    _provider, manager = provider_env
    _force_policy(monkeypatch, False)
    agent = _agent(manager, bridge=True)

    added = mm.inject_memory_provider_tools(agent)

    attached = {t["function"]["name"] for t in agent.tools if isinstance(t, dict)}
    assert added == 4
    assert set(ALL_TOOLS) <= attached
    assert mm.deferred_provider_tool_names() == set()


def test_injection_full_eager_without_bridge(provider_env, monkeypatch):
    """Deferral without a bridge would hide tools with no way to reach them."""
    _provider, manager = provider_env
    _force_policy(monkeypatch, True)
    agent = _agent(manager, bridge=False)

    added = mm.inject_memory_provider_tools(agent)

    attached = {t["function"]["name"] for t in agent.tools if isinstance(t, dict)}
    assert added == 4
    assert set(ALL_TOOLS) <= attached


def test_classification_follows_deferral(provider_env, monkeypatch):
    _provider, manager = provider_env
    _force_policy(monkeypatch, True)
    mm.register_provider_tool_defs(_provider.get_tool_schemas())
    from tools.tool_search import is_deferrable_tool_name

    assert is_deferrable_tool_name("mem_sleep") is True
    assert is_deferrable_tool_name("mem_export") is True
    assert is_deferrable_tool_name("mem_recall") is False  # eager: direct-call only
    _force_policy(monkeypatch, False)
    assert is_deferrable_tool_name("mem_sleep") is False


def test_bridge_defs_gate_on_memory_toolset(provider_env, monkeypatch):
    _provider, _manager = provider_env
    _force_policy(monkeypatch, True)
    mm.register_provider_tool_defs(_provider.get_tool_schemas())

    defs = mm.deferred_provider_tool_defs(None, None, memory_tool_present=True)
    assert {d["function"]["name"] for d in defs} == {"mem_sleep", "mem_export"}
    # memory toolset gated off → nothing rides the bridge either
    assert mm.deferred_provider_tool_defs(["core"], None, memory_tool_present=False) == []
    assert mm.deferred_provider_tool_defs(None, ["memory"], memory_tool_present=True) == []


def test_bridge_tool_call_executes_via_memory_manager(provider_env, monkeypatch):
    provider, manager = provider_env
    _force_policy(monkeypatch, True)
    mm.register_provider_tool_defs(provider.get_tool_schemas())

    from agent import tool_executor as te
    monkeypatch.setattr(te, "_start_quiet_tool_spinner", lambda *a, **k: None)
    agent = SimpleNamespace(_memory_manager=manager, _context_engine_tool_names=set())
    ref = te._ToolCallRef(
        name="tool_call",
        args={"name": "mem_sleep", "arguments": {"force": True}},
        task_id="t-1", call_id="c-1", trace=[],
    )

    dispatch = te._resolve_sequential_dispatch(agent, ref, [])
    result = dispatch.execute(ref.args)

    assert json.loads(result)["handled"] == "mem_sleep"
    assert provider.calls == [("mem_sleep", {"force": True})]


def test_bridge_tool_call_for_registry_tool_untouched(provider_env, monkeypatch):
    """A bridge call that is NOT a provider tool keeps the normal registry path."""
    provider, manager = provider_env
    _force_policy(monkeypatch, True)
    mm.register_provider_tool_defs(provider.get_tool_schemas())

    from agent import tool_executor as te
    monkeypatch.setattr(te, "_start_quiet_tool_spinner", lambda *a, **k: None)
    called = {}

    def _fake_handle(*a, **k):
        called["hit"] = True
        return '{"ok": true}'

    import model_tools
    monkeypatch.setattr(model_tools, "handle_function_call", _fake_handle)
    agent = SimpleNamespace(_memory_manager=manager, _context_engine_tool_names=set(),
                            valid_tool_names={"tool_call"}, enabled_toolsets=None,
                            disabled_toolsets=None, quiet_mode=True, log_prefix_chars=80,
                            _tool_call_observers=[], tool_progress_callback=None,
                            _current_tool=None, _touch_activity=lambda *a, **k: None,
                            _set_worker_activity_callback=None)
    ref = te._ToolCallRef(name="tool_call", args={"name": "some_registry_tool", "arguments": {}},
                          task_id="t-1", call_id="c-1", trace=[])
    dispatch = te._resolve_sequential_dispatch(agent, ref, [])
    # memory manager must NOT have been asked to handle a non-provider tool
    assert provider.calls == []
    assert dispatch is not None


def test_bridge_merge_dedupes_registry_plugin_copies(provider_env, monkeypatch):
    """Registry-side copies of provider verbs win; provider defs only fill gaps.

    Production shape: the memory plugin registers its verbs as registry plugin
    tools AND the deferral store publishes the same schemas (kanban t_a86c18b8).
    Without the dedupe the bridge universe lists every verb twice — duplicate
    search hits, inflated total_available, collapse sizing on a double count.
    """
    provider, manager = provider_env
    _force_policy(monkeypatch, True)
    mm.register_provider_tool_defs(provider.get_tool_schemas())

    import model_tools

    registry_copy = {"type": "function",
                     "function": {"name": "mem_sleep", "description": "registry copy",
                                  "parameters": {"type": "object", "properties": {}}}}
    base = [{"type": "function", "function": {"name": "memory"}}, registry_copy]
    merged = model_tools._merge_deferred_provider_defs(base, None, None)

    names = [(t.get("function") or {}).get("name") for t in merged]
    assert names.count("mem_sleep") == 1
    assert "mem_export" in names  # provider-only name still merged
    kept = next(t for t in merged if (t.get("function") or {}).get("name") == "mem_sleep")
    assert kept is registry_copy  # existing copy wins (scope/validation consistency)
    assert len(base) == 2  # input list not mutated


def test_bridge_merge_noop_when_policy_off(provider_env, monkeypatch):
    provider, manager = provider_env
    _force_policy(monkeypatch, False)
    mm.register_provider_tool_defs(provider.get_tool_schemas())

    import model_tools
    base = [{"type": "function", "function": {"name": "memory"}}]
    assert model_tools._merge_deferred_provider_defs(base, None, None) is base


def test_bridge_merge_noop_when_memory_toolset_disabled(provider_env, monkeypatch):
    provider, manager = provider_env
    _force_policy(monkeypatch, True)
    mm.register_provider_tool_defs(provider.get_tool_schemas())

    import model_tools
    base = [{"type": "function", "function": {"name": "memory"}}]
    merged = model_tools._merge_deferred_provider_defs(base, None, ["memory"])
    assert merged == base  # memory toolset disabled → nothing merges


def test_skills_index_styles():
    from agent import prompt_builder as pb

    long_desc = "Use when a very long explanation. " + ("filler words " * 40)
    cats = {"demo": [("sk_short", "Short one."), ("sk_long", long_desc)]}
    full = pb._render_skills_index(cats, {}, None, None)
    trig = pb._render_skills_index(cats, {}, None, None, index_style="trigger")
    names = pb._render_skills_index(cats, {}, None, None, index_style="names")

    assert long_desc in full
    long_line = next(l for l in trig.splitlines() if "sk_long" in l)
    assert len(long_line) < len(long_desc) + 20
    assert long_line.rstrip().endswith(("words", "when a very long explanation."))
    assert "sk_long:" not in names
    assert "[names only]: sk_long, sk_short" in names  # alphabetical, nothing dropped
    assert "keep the prompt compact" in names  # the disclosure note stays


def test_compact_skill_desc_deterministic():
    from agent import prompt_builder as pb

    desc = "Use when diagnosing faults. Extra sentence one. Extra sentence two."
    a = pb._compact_skill_desc(desc, cap=40)
    b = pb._compact_skill_desc(desc, cap=40)
    assert a == b
    assert len(a) <= 40
    assert pb._compact_skill_desc("short", cap=40) == "short"
    assert pb._compact_skill_desc("  padded   spacing  ", cap=40) == "padded spacing"


def test_policy_cache_is_scope_keyed(provider_env, monkeypatch):
    """Regression (t_eb99c448 live postmortem): one process-wide policy slot let a
    staff-profile agent init (config WITHOUT the deferral block) poison the default
    profile's Fleet injects for the whole TTL — the multiplex gateway kept shipping
    all 41 provider tools. Per-config-path keys must never cross-contaminate."""
    import hermes_cli.config as hc

    scope = {"cur": "staffA"}
    cfgs = {
        "staffA": {"memory": {}},  # no provider_tool_deferral block at all
        "default": {"memory": {"provider_tool_deferral": {
            "enabled": True, "eager_tools": list(EAGER)}}},
    }
    monkeypatch.setattr(mm, "_policy_scope_key", lambda: scope["cur"])
    monkeypatch.setattr(hc, "load_config_readonly", lambda: cfgs[scope["cur"]])
    mm._policy_cache.clear()
    mm._deferred_names_cache.clear()

    scope["cur"] = "staffA"
    assert mm.provider_tool_deferral_policy()["enabled"] is False
    scope["cur"] = "default"  # WITHIN the TTL of the staffA read
    pol = mm.provider_tool_deferral_policy()
    assert pol["enabled"] is True
    assert pol["eager_tools"] == EAGER
    scope["cur"] = "staffA"  # and the staff entry stayed False, not overwritten
    assert mm.provider_tool_deferral_policy()["enabled"] is False


def test_provider_defs_generation_bumps_only_on_change(provider_env):
    provider, _ = provider_env
    mm.register_provider_tool_defs(provider.get_tool_schemas())
    g1 = mm.provider_defs_generation()
    assert g1 >= 1
    mm.register_provider_tool_defs(provider.get_tool_schemas())  # identical content
    assert mm.provider_defs_generation() == g1
    mm.register_provider_tool_defs([_schema("mem_sleep", desc="changed")])
    assert mm.provider_defs_generation() == g1 + 1


def test_defs_cache_key_tracks_provider_generation(provider_env):
    """The tool-defs memo must not pin a pre-registration resolve: once the provider
    publishes its schemas, the merged disclosure assembly differs (catalog + search
    gain the deferred tail) and the next resolve has to recompute."""
    import model_tools as mt

    provider, _ = provider_env
    k1 = mt._tool_defs_cache_key(None, None, False)
    assert k1 is not None
    mm.register_provider_tool_defs(provider.get_tool_schemas())
    k2 = mt._tool_defs_cache_key(None, None, False)
    assert k1 != k2
    mm.register_provider_tool_defs(provider.get_tool_schemas())  # no content change
    assert mt._tool_defs_cache_key(None, None, False) == k2


def test_activation_path_publishes_defs(provider_env, monkeypatch):
    """plugins/memory's _ProviderCollector is how the GATEWAY loads providers; it must
    publish to the discovery store like PluginContext does, or bridge catalog/search
    and tool_call unwrap never see the deferred tail in the gateway process."""
    from plugins.memory import _ProviderCollector

    provider, _ = provider_env
    mm.register_provider_tool_defs([])  # store starts empty
    collector = _ProviderCollector("memtest")
    collector.register_memory_provider(provider)
    assert mm.provider_defs_generation() >= 1
    assert set(mm._PROVIDER_TOOL_DEFS) == set(ALL_TOOLS)

    # inspection mode (register_skills=False) must NOT publish
    monkeypatch.setattr(mm, "_PROVIDER_TOOL_DEFS", {})
    monkeypatch.setattr(mm, "_PROVIDER_DEFS_GENERATION", 0)
    inspector = _ProviderCollector("memtest", register_skills=False)
    inspector.register_memory_provider(provider)
    assert mm._PROVIDER_TOOL_DEFS == {}


def test_rebuild_reinject_defers_cold_tail_when_policy_on(provider_env, monkeypatch):
    """Regression (t_eb99c448 second fault): the between-turns rebuild re-appended
    EVERY provider verb onto the staged wire, silently re-fattening each live Fleet
    turn back to 91 tools / 41,633 first-call tokens even while init-time injection
    correctly deferred. The rebuild mirror must attach only the eager pair and hand
    the deferred names back so they stay valid tool names."""
    _provider, manager = provider_env
    _force_policy(monkeypatch, True)
    agent = _agent(manager, bridge=True)

    from tools.mcp_tool_agent import _reinject_post_build_tools
    staged = [{"type": "function", "function": {"name": "memory"}},
              {"type": "function", "function": {"name": "tool_search"}}]
    names = {"memory", "tool_search"}
    engine_names, deferred = _reinject_post_build_tools(agent, staged, names)

    wire = {t["function"]["name"] for t in staged}
    assert wire == {"memory", "tool_search", *EAGER}
    assert deferred == set(ALL_TOOLS) - set(EAGER)
    assert engine_names == set()


def test_rebuild_reinject_full_attach_when_policy_off(provider_env, monkeypatch):
    _provider, manager = provider_env
    _force_policy(monkeypatch, False)
    agent = _agent(manager, bridge=True)

    from tools.mcp_tool_agent import _reinject_post_build_tools
    staged = [{"type": "function", "function": {"name": "memory"}},
              {"type": "function", "function": {"name": "tool_search"}}]
    _engine, deferred = _reinject_post_build_tools(agent, staged, {"memory", "tool_search"})
    assert {t["function"]["name"] for t in staged} == {"memory", "tool_search", *ALL_TOOLS}
    assert deferred == set()


def test_rebuild_reinject_full_attach_without_bridge(provider_env, monkeypatch):
    """No bridge staged → nothing to defer behind: classic full attach, the same
    contract inject_memory_provider_tools honors."""
    _provider, manager = provider_env
    _force_policy(monkeypatch, True)
    agent = _agent(manager, bridge=False)

    from tools.mcp_tool_agent import _reinject_post_build_tools
    staged = [{"type": "function", "function": {"name": "memory"}}]
    _engine, deferred = _reinject_post_build_tools(agent, staged, {"memory"})
    assert {t["function"]["name"] for t in staged} == {"memory", *ALL_TOOLS}
    assert deferred == set()
