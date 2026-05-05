"""Integration tests for piaware-local-flights plugin."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
from PIL import Image, ImageDraw, ImageFont

from test.plugins.test_plugin_base import PluginTestBase


class _StubDisplayManager:
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
    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._payload


class TestPiAwareLocalFlightsPlugin(PluginTestBase):
    @pytest.fixture
    def plugin_id(self):
        return "piaware-local-flights"

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

    def test_plugin_display_method(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.display(force_clear=True)
        assert hasattr(plugin, "display")

    def test_empty_state_without_aircraft(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(200, {"now": 1000, "aircraft": []})

        plugin.update()

        assert plugin.status_code == "empty"
        assert plugin.current_flight is None

    def test_selects_nearest_aircraft(self, plugin_id):
        display = _StubDisplayManager()
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
            "show_altitude": True,
            "show_speed": True,
        }
        plugin = self._instantiate_plugin(plugin_id, config, display)
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "now": 1000,
                "aircraft": [
                    {"hex": "A1B2C3", "flight": "UAL123", "lat": 33.70, "lon": -116.32, "alt_baro": 35000, "gs": 420, "seen": 2},
                    {"hex": "D4E5F6", "flight": "DAL456", "lat": 34.10, "lon": -116.80, "alt_baro": 20000, "gs": 320, "seen": 2},
                ],
            },
        )

        plugin.update()
        plugin.display(force_clear=True)

        assert plugin.status_code == "ok"
        assert plugin.current_flight["callsign"] == "UAL123"
        assert plugin.current_flight["airline_code"] == "UAL"
        assert plugin.current_flight["altitude_ft"] == 35000.0
        assert any(pixel != (0, 0, 0) for pixel in display.image.getdata())

    def test_route_text_uses_enriched_origin_destination_when_available(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())

        text = plugin._route_like_text(
            {
                "origin": "LAX",
                "destination": "ORD",
                "direction_home": "NE",
                "distance_km": 12.0,
            }
        )

        assert text == "LAX->ORD"

    def test_route_text_falls_back_without_enrichment(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())

        text = plugin._route_like_text(
            {
                "direction_home": "NE",
                "distance_km": 12.0,
            }
        )

        assert text == "NE 12km"

    def test_commercial_only_filters_private_aircraft(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
            "commercial_only": True,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "now": 1000,
                "aircraft": [
                    {"hex": "ABC123", "flight": "N123AB", "r": "N123AB", "lat": 33.70, "lon": -116.32, "seen": 2},
                    {"hex": "A1B2C3", "flight": "UAL123", "lat": 33.71, "lon": -116.31, "alt_baro": 35000, "gs": 420, "seen": 2},
                ],
            },
        )

        plugin.update()

        assert plugin.status_code == "ok"
        assert len(plugin.current_aircraft) == 1
        assert plugin.current_flight["callsign"] == "UAL123"

    def test_commercial_only_empty_message(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
            "commercial_only": True,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "now": 1000,
                "aircraft": [
                    {"hex": "ABC123", "flight": "N123AB", "r": "N123AB", "lat": 33.70, "lon": -116.32, "seen": 2},
                ],
            },
        )

        plugin.update()

        assert plugin.status_code == "empty"
        assert plugin.status_message == "No airline"

    def test_reads_local_receiver_json_file(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        payload = {
            "now": 1000,
            "aircraft": [
                {"hex": "A1B2C3", "flight": "UAL123", "lat": 33.70, "lon": -116.32, "alt_baro": 35000, "gs": 420, "seen": 2}
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            temp_path = handle.name

        plugin.receiver_urls = [temp_path]
        plugin.update()

        assert plugin.status_code == "ok"
        assert plugin.current_flight["callsign"] == "UAL123"

    def test_filters_stale_aircraft(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
            "max_seen_seconds": 30,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "now": 1000,
                "aircraft": [
                    {"hex": "A1B2C3", "flight": "UAL123", "lat": 33.70, "lon": -116.32, "seen": 120},
                ],
            },
        )

        plugin.update()

        assert plugin.status_code == "empty"

    def test_receiver_error_uses_cache(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
            "cache_seconds": 10,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        cached = [
            {
                "hex": "A1B2C3",
                "callsign": "UAL123",
                "distance_km": 12.0,
                "direction_home": "NE",
                "aircraft_type": "A320",
                "airline_code": "UAL",
            }
        ]
        plugin._cache_flights(cached)

        def raise_error(*args, **kwargs):
            raise requests.ConnectionError("receiver offline")

        plugin.last_update = 0.0
        plugin._load_cached_flights = lambda max_age: [] if max_age <= plugin.cache_seconds else cached
        plugin.session.get = raise_error
        plugin.update()

        assert plugin.status_code == "receiver_error"
        assert plugin.current_flight["callsign"] == "UAL123"

    def test_cached_fr24_enrichment_is_applied(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
            "fr24_enrichment_enabled": True,
            "fr24_enrichment_cache_seconds": 1800,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.cache_manager.set(
            plugin.fr24_enrichment_cache_key,
            {
                "A1B2C3": {
                    "origin": "LAX",
                    "destination": "ORD",
                    "aircraft_type": "A319",
                    "airline_code": "UAL",
                }
            },
            ttl=plugin.fr24_enrichment_cache_seconds,
        )
        plugin.session.get = lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "now": 1000,
                "aircraft": [
                    {"hex": "A1B2C3", "flight": "UAL123", "lat": 33.70, "lon": -116.32, "alt_baro": 35000, "gs": 420, "seen": 2}
                ],
            },
        )

        plugin.update()

        assert plugin.current_flight["origin"] == "LAX"
        assert plugin.current_flight["destination"] == "ORD"
        assert plugin.current_flight["aircraft_type"] == "A319"

    def test_logo_uses_led_directory_first(self, plugin_id):
        config = {
            **self.base_config,
            "enabled": True,
            "lat": 33.6634,
            "lon": -116.3100,
            "radius_km": 120,
        }
        plugin = self._instantiate_plugin(plugin_id, config, _StubDisplayManager())
        plugin.current_flight = {"airline_code": "UAL"}
        seen_paths = []

        def fake_load_logo(_code, logo_path, max_width, max_height):
            seen_paths.append(Path(logo_path))
            return None

        plugin.logo_helper.load_logo = fake_load_logo
        plugin._load_airline_logo(plugin.current_flight, plugin.display_manager.height)

        assert seen_paths[0] == Path("assets/airline_logos_piaware/UAL.png")
