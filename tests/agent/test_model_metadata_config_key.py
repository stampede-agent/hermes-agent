"""Tests for _config_api_key_for_base_url and the WS6 key-resolution wiring in
the metadata probes (kanban t_5936f60e): anonymous probes against an endpoint
the local config holds a key for produced a 401 + traceback on the fleet router
for every fresh hermes process ("GET /v1/models every ~5 min" bursts). Callers
that pass no api_key must now use the configured key for that origin.

unittest-style so it runs under `python -m unittest` as well as pytest.
"""

import unittest
from unittest.mock import patch

import agent.model_metadata as mm

FAKE_KEY = "fleet…-key"
FAKE_CONFIG = {
    "model": {
        "aliases": {
            "auto": {
                "model": "auto", "provider": "custom",
                "base_url": "http://127.0.0.1:4000/v1", "api_key": FAKE_KEY,
            },
            # entry without a key must not shadow the keyed one for the origin
            "cluster": {"model": "cluster-big", "base_url": "http://127.0.0.1:4000/v1"},
        },
    },
    "providers": {
        "spark": {"base_url": "http://10.0.69.29:11434/v1", "api_key": ""},
    },
}


class _MemoClearMixin(unittest.TestCase):
    def setUp(self):
        mm._config_key_memo.clear()

    def tearDown(self):
        mm._config_key_memo.clear()


class TestOriginKey(_MemoClearMixin):
    def test_default_ports(self):
        self.assertEqual(mm._origin_key("https://example.com/v1"), "https://example.com:443")
        self.assertEqual(mm._origin_key("http://example.com"), "http://example.com:80")

    def test_localhost_spellings_collapse(self):
        self.assertEqual(mm._origin_key("http://localhost:4000/v1"),
                         mm._origin_key("http://127.0.0.1:4000/v1"))
        self.assertEqual(mm._origin_key("http://[::1]:4000/v1"),
                         mm._origin_key("http://127.0.0.1:4000/v1"))

    def test_garbage_in_empty_out(self):
        self.assertEqual(mm._origin_key(""), "")
        self.assertEqual(mm._origin_key("not a url"), "")


class TestConfigApiKeyLookup(_MemoClearMixin):
    def test_alias_match_ignores_path_and_host_spelling(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG):
            self.assertEqual(mm._config_api_key_for_base_url("http://127.0.0.1:4000/v1"), FAKE_KEY)
        mm._config_key_memo.clear()
        with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG):
            self.assertEqual(mm._config_api_key_for_base_url("http://localhost:4000"), FAKE_KEY)

    def test_unknown_origin_returns_empty(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG):
            self.assertEqual(mm._config_api_key_for_base_url("http://127.0.0.1:9999/v1"), "")

    def test_empty_key_entry_returns_empty(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG):
            self.assertEqual(mm._config_api_key_for_base_url("http://10.0.69.29:11434/v1"), "")

    def test_memoized_per_origin(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG) as m:
            mm._config_api_key_for_base_url("http://127.0.0.1:4000/v1")
            mm._config_api_key_for_base_url("http://127.0.0.1:4000/v1")
            self.assertEqual(m.call_count, 1)

    def test_config_errors_degrade_to_empty(self):
        with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
            self.assertEqual(mm._config_api_key_for_base_url("http://127.0.0.1:4000/v1"), "")


def _fake_stream_capture(seen):
    def fake_stream(url, headers=None, timeout=None, verify=None):
        seen["url"] = url
        seen["headers"] = headers or {}
        raise RuntimeError("stop-after-headers")
    return fake_stream


def _fetch_with_mocks(base_url, api_key=""):
    seen = {}
    with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG), \
         patch.object(mm.model_metadata_http, "stream", side_effect=_fake_stream_capture(seen)), \
         patch.object(mm, "_endpoint_blackholed", return_value=False), \
         patch.object(mm, "_endpoint_disk_cache_get", return_value=None), \
         patch.object(mm, "detect_local_server_type", return_value=None):
        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()
        out = mm.fetch_endpoint_model_metadata(base_url, api_key=api_key)
    return out, seen


class TestFetchEndpointWiring(_MemoClearMixin):
    def test_keyless_fetch_uses_configured_key(self):
        out, seen = _fetch_with_mocks("http://127.0.0.1:4000/v1")
        self.assertEqual(out, {})
        self.assertEqual(seen["headers"].get("Authorization"), f"Bearer {FAKE_KEY}")

    def test_explicit_key_wins_over_config(self):
        _out, seen = _fetch_with_mocks("http://127.0.0.1:4000/v1", api_key="explicit-test-key")
        self.assertEqual(seen["headers"].get("Authorization"), "Bearer explicit-test-key")

    def test_unknown_origin_stays_anonymous(self):
        _out, seen = _fetch_with_mocks("http://127.0.0.1:9999/v1")
        self.assertNotIn("Authorization", seen["headers"])


class TestDetectLocalServerTypeWiring(_MemoClearMixin):
    def test_waterfall_gets_configured_key(self):
        captured = {}

        class FakeClient:
            def __init__(self, timeout=None, headers=None, verify=None):
                captured["headers"] = headers or {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url):
                raise RuntimeError("stop")

        mm._endpoint_probe_path_cache.clear()
        try:
            with patch("hermes_cli.config.load_config_readonly", return_value=FAKE_CONFIG), \
                 patch.object(mm, "_local_probe_disk_get", return_value=None), \
                 patch.object(mm, "_endpoint_blackholed", return_value=False), \
                 patch("httpx.Client", FakeClient):
                mm.detect_local_server_type("http://127.0.0.1:4000/v1")
            self.assertEqual(captured["headers"].get("Authorization"), f"Bearer {FAKE_KEY}")
        finally:
            mm._endpoint_probe_path_cache.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
