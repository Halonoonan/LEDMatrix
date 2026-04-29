"""
Integration tests for flightradar24 plugin.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import requests
from PIL import Image, ImageDraw, ImageFont

from test.plugins.test_plugin_base import PluginTestBase


class _StubDisplayManager:
    """Small real-image display surface for rendering assertions."""

    def __init__(self, width: int = 128, height: int = 32) -> None:
        self.matrix = type("Matrix", (), {"width": width, "height": height})()
        self.image = Image.new("RGB", (width, height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.extra_small_font = ImageFont.load_default()
        self.small_font = ImageFont.load_default()

    @property
    def width(self) -> int:
        return self.matrix.width

    @property
    def height(self) -> int:
        return self.matrix.height

    def clear(self) -> None:
        self.image = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def update_display(self) -> None:
        return

    def get_font_height(self, font: Any) -> int:
        try:
            bbox = font.getbbox("Ag")
            return bbox[3] - bbox[1]
        except Exception:
            return 8


class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> Dict[str, Any]:
        return self._payload


class TestFlightRadar24Plugin(PluginTestBase):
    """Test flightradar24 plugin integration and mocked API paths."""

    @pytest.fixture
    def plugin_id(self):
        return "flightradar24"

    def test_manifest_exists(self, plugin_id):
        super().test_manifest_exists(plugin_id)

    def test_manifest_has_required_fields(self, plugin_id):
        super().test_manifest_has_required_fields(plugin_id)

    def test_plugin_can_be_loaded(self, plugin_id):
        super().test_plugin_can_be_loaded(plugin_id)

    def test_plugin_class_exists(self, plugin_id):
        super().test_plugin_class_exists(plugin_id)

    def test_plugin_can_be_instantiated(self, plugin_id):
        super().test_plugin_can_be_instantiated(plugin_id)

    def test_plugin_has_required_methods(self, plugin_id):
        super().test_plugin_has_required_methods(plugin_id)

    def test_plugin_display_method(self, plugin_id):
        """Use a real display stub because this plugin measures font height."""
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())

        plugin.update()
        plugin.display(force_clear=True)

        assert hasattr(plugin, "display")

    def test_plugin_has_display_modes(self, plugin_id):
        manifest = self.load_plugin_manifest(plugin_id)
        assert "display_modes" in manifest
        assert "flightradar24" in manifest["display_modes"]

    def test_config_schema_valid(self, plugin_id):
        super().test_config_schema_valid(plugin_id)

    def _instantiate_plugin(self, plugin_id: str, config: Dict[str, Any], display_manager: Any):
        manifest = self.load_plugin_manifest(plugin_id)
        plugin_dir = self.plugins_dir / plugin_id
        module = self.plugin_loader.load_module(
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            entry_point=manifest.get("entry_point", "manager.py"),
        )
        plugin_class = self.plugin_loader.get_plugin_class(
            plugin_id=plugin_id,
            module=module,
            class_name=manifest["class_name"],
        )
        return self.plugin_loader.instantiate_plugin(
            plugin_id=plugin_id,
            plugin_class=plugin_class,
            config=config,
            display_manager=display_manager,
            cache_manager=self.mock_cache_manager,
            plugin_manager=self.mock_plugin_manager,
        )

    def test_missing_token_state(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "primary_color": [255, 255, 255],
            "secondary_color": [0, 255, 255],
            "show_aircraft_type": False,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())

        plugin.update()

        assert plugin.status_code == "missing_token"
        assert plugin.current_flight is None

    def test_mocked_success_and_display_render(self, plugin_id):
        display = _StubDisplayManager()
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "primary_color": [255, 255, 255],
            "secondary_color": [0, 255, 255],
            "show_aircraft_type": False,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, display)

        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "data": [
                    {
                        "callsign": "UAL123",
                        "orig_iata": "SFO",
                        "dest_iata": "LAX",
                        "type": "B739",
                        "lat": 33.7000,
                        "lon": -116.3200,
                    },
                    {
                        "callsign": "AAL456",
                        "orig_iata": "PHX",
                        "dest_iata": "SAN",
                        "type": "A320",
                        "lat": 33.9000,
                        "lon": -116.4500,
                    },
                ]
            },
        )

        plugin.update()
        plugin.display(force_clear=True)

        assert plugin.status_code == "ok"
        assert len(plugin.current_flights) == 2
        assert plugin.current_flight["callsign"] == "UAL123"
        assert any(pixel != (0, 0, 0) for pixel in display.image.getdata())

    def test_mocked_nested_route_fields(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        self.mock_cache_manager._memory_cache.clear()
        plugin.last_update = 0.0
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "data": [
                    {
                        "flight": {"callsign": "AIK594"},
                        "aircraft": {"model": {"code": "A21N"}},
                        "airport": {
                            "origin": {"code": {"iata": "MIA", "icao": "KMIA"}},
                            "destination": {"code": {"iata": "PSP", "icao": "KPSP"}},
                        },
                        "lat": 33.7000,
                        "lon": -116.3200,
                    }
                ]
            },
        )

        plugin.update()

        assert plugin.status_code == "ok"
        assert plugin.current_flight["callsign"] == "AIK594"
        assert plugin.current_flight["origin"] == "MIA"
        assert plugin.current_flight["destination"] == "PSP"
        assert plugin.current_flight["aircraft_type"] == "A21N"

    def test_prefers_full_endpoint(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        self.mock_cache_manager._memory_cache.clear()
        plugin.last_update = 0.0
        calls = []

        def fake_get(url, *args, **kwargs):
            calls.append(url)
            return _FakeResponse(
                200,
                {
                    "data": [
                        {
                            "callsign": "UAL123",
                            "airport": {
                                "origin": {"code": {"iata": "SFO"}},
                                "destination": {"code": {"iata": "LAX"}},
                            },
                            "lat": 33.7000,
                            "lon": -116.3200,
                        }
                    ]
                },
            )

        plugin.session.get = fake_get
        plugin.update()

        assert calls[0].endswith("/live/flight-positions/full")
        assert plugin.current_flight["origin"] == "SFO"
        assert plugin.current_flight["destination"] == "LAX"

    def test_falls_back_to_light_endpoint(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        self.mock_cache_manager._memory_cache.clear()
        plugin.last_update = 0.0
        calls = []

        def fake_get(url, *args, **kwargs):
            calls.append(url)
            if url.endswith("/live/flight-positions/full"):
                return _FakeResponse(403, {})
            return _FakeResponse(
                200,
                {
                    "data": [
                        {
                            "callsign": "SWA839",
                            "lat": 33.60862,
                            "lon": -116.55819,
                        }
                    ]
                },
            )

        plugin.session.get = fake_get
        plugin.update()

        assert calls[0].endswith("/live/flight-positions/full")
        assert calls[1].endswith("/live/flight-positions/light")
        assert plugin.status_code == "ok"
        assert plugin.current_flight["callsign"] == "SWA839"

    @pytest.mark.parametrize(
        ("status_code", "headers", "expected_status"),
        [
            (401, {}, "invalid_token"),
            (403, {}, "endpoint_access"),
            (429, {"Retry-After": "60"}, "rate_limited"),
        ],
    )
    def test_mocked_http_error_states(self, plugin_id, status_code, headers, expected_status):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        self.mock_cache_manager._memory_cache.clear()
        plugin.last_update = 0.0
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(status_code, {}, headers)

        plugin.update()

        assert plugin.status_code == expected_status

    def test_mocked_empty_result_state(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        self.mock_cache_manager._memory_cache.clear()
        plugin.last_update = 0.0
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(200, {"data": []})

        plugin.update()

        assert plugin.status_code == "empty"
        assert plugin.current_flight is None

    def test_mocked_network_exception_state(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 40,
            "cache_seconds": 30,
            "display_duration": 12,
            "api_token": "demo-token",
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        self.mock_cache_manager._memory_cache.clear()
        plugin.last_update = 0.0

        def raise_network_error(*args, **kwargs):
            raise requests.exceptions.ConnectionError("mock network down")

        plugin.session.get = raise_network_error
        plugin.update()

        assert plugin.status_code == "network_error"
        assert plugin.current_flight is None
