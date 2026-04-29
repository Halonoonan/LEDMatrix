"""FlightRadar24 nearby flights plugin."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.plugin_system.base_plugin import BasePlugin


class FlightRadar24Plugin(BasePlugin):
    """Display the nearest live flight around a configured home location."""

    API_ENV_VARS = (
        "LEDMATRIX_FLIGHTRADAR24_API_TOKEN",
        "FR24_API_TOKEN",
        "FR24_TOKEN",
    )

    def __init__(
        self,
        plugin_id: str,
        config: Dict[str, Any],
        display_manager: Any,
        cache_manager: Any,
        plugin_manager: Any,
    ) -> None:
        super().__init__(plugin_id, config, display_manager, cache_manager, plugin_manager)
        self._apply_config(config)
        self.session = self._build_session()
        self.current_flights: List[Dict[str, Any]] = []
        self.current_flight: Optional[Dict[str, Any]] = None
        self.status_code = "init"
        self.status_message = "Starting"
        self.last_update = 0.0

    def _apply_config(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.lat = float(self.config.get("lat", 0.0))
        self.lon = float(self.config.get("lon", 0.0))
        self.radius_km = float(self.config.get("radius_km", 40))
        self.cache_seconds = max(10, int(self.config.get("cache_seconds", 30)))
        self.display_duration = float(self.config.get("display_duration", 12))
        self.primary_color = self._normalize_color(self.config.get("primary_color"), (255, 255, 255))
        self.secondary_color = self._normalize_color(self.config.get("secondary_color"), (0, 255, 255))
        self.show_aircraft_type = bool(self.config.get("show_aircraft_type", False))
        self.api_base_url = str(
            self.config.get("api_base_url", "https://fr24api.flightradar24.com/api")
        ).rstrip("/")
        self.request_timeout = max(3, int(self.config.get("request_timeout", 10)))
        self.result_limit = max(1, min(100, int(self.config.get("result_limit", 20))))
        endpoint_order = self.config.get("endpoint_order", ["full", "light"])
        if not isinstance(endpoint_order, list):
            endpoint_order = ["full", "light"]
        self.endpoint_order = [
            str(item).strip().lower()
            for item in endpoint_order
            if str(item).strip().lower() in {"full", "light"}
        ] or ["full", "light"]
        self.api_token = self._resolve_api_token()
        self.cache_key = (
            f"{self.plugin_id}_{self.lat:.4f}_{self.lon:.4f}_"
            f"{self.radius_km:.1f}_{self.result_limit}"
        )

    @staticmethod
    def _normalize_color(value: Any, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return tuple(max(0, min(255, int(channel))) for channel in value)
            except (TypeError, ValueError):
                return fallback
        return fallback

    def _resolve_api_token(self) -> str:
        token = str(self.config.get("api_token", "") or "").strip()
        if token:
            return token
        for env_name in self.API_ENV_VARS:
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                return env_value
        return ""

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _build_bounds(self) -> str:
        lat_delta = self.radius_km / 111.32
        cos_lat = math.cos(math.radians(self.lat))
        lon_delta = self.radius_km / (111.32 * max(abs(cos_lat), 0.01))
        north = min(90.0, self.lat + lat_delta)
        south = max(-90.0, self.lat - lat_delta)
        west = max(-180.0, self.lon - lon_delta)
        east = min(180.0, self.lon + lon_delta)
        return f"{north:.4f},{south:.4f},{west:.4f},{east:.4f}"

    def _request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Accept-Version": "v1",
            "User-Agent": "LEDMatrix/FlightRadar24Plugin",
        }

    @staticmethod
    def _coerce_float(*values: Any) -> Optional[float]:
        for value in values:
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coerce_str(*values: Any, fallback: str = "") -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple, set)):
                continue
            text = str(value).strip()
            if text:
                return text
        return fallback

    def _extract_airport_code(self, airport_data: Any) -> str:
        """Pull the most useful short airport code from varying FR24 shapes."""
        if not isinstance(airport_data, dict):
            return ""

        code = airport_data.get("code") if isinstance(airport_data.get("code"), dict) else {}
        info = airport_data.get("info") if isinstance(airport_data.get("info"), dict) else {}

        return self._coerce_str(
            airport_data.get("iata"),
            airport_data.get("fs"),
            airport_data.get("icao"),
            airport_data.get("iataCode"),
            airport_data.get("icaoCode"),
            code.get("iata"),
            code.get("icao"),
            code.get("iataCode"),
            code.get("icaoCode"),
            info.get("iata"),
            info.get("icao"),
        )

    def _extract_flights(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "flights", "items", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    def _parse_route(self, raw_flight: Dict[str, Any]) -> Tuple[str, str]:
        route = raw_flight.get("route") if isinstance(raw_flight.get("route"), dict) else {}
        airport = raw_flight.get("airport") if isinstance(raw_flight.get("airport"), dict) else {}
        origin = route.get("origin") if isinstance(route.get("origin"), dict) else {}
        destination = route.get("destination") if isinstance(route.get("destination"), dict) else {}
        airport_origin = airport.get("origin") if isinstance(airport.get("origin"), dict) else {}
        airport_destination = airport.get("destination") if isinstance(airport.get("destination"), dict) else {}
        airport_code = airport.get("code") if isinstance(airport.get("code"), dict) else {}

        orig = self._coerce_str(
            raw_flight.get("orig_iata"),
            raw_flight.get("orig_icao"),
            raw_flight.get("origin_iata"),
            raw_flight.get("origin_icao"),
            raw_flight.get("airport_origin"),
            raw_flight.get("origin"),
            route.get("origin"),
            route.get("from"),
            route.get("departure"),
            route.get("originIata"),
            route.get("originIcao"),
            origin.get("iata"),
            origin.get("icao"),
            origin.get("code"),
            self._extract_airport_code(origin),
            self._extract_airport_code(airport_origin),
            airport_code.get("origin"),
            airport.get("origin"),
            fallback="???",
        )
        dest = self._coerce_str(
            raw_flight.get("dest_iata"),
            raw_flight.get("dest_icao"),
            raw_flight.get("destination_iata"),
            raw_flight.get("destination_icao"),
            raw_flight.get("airport_destination"),
            raw_flight.get("destination"),
            route.get("destination"),
            route.get("to"),
            route.get("arrival"),
            route.get("destinationIata"),
            route.get("destinationIcao"),
            destination.get("iata"),
            destination.get("icao"),
            destination.get("code"),
            self._extract_airport_code(destination),
            self._extract_airport_code(airport_destination),
            airport_code.get("destination"),
            airport.get("destination"),
            fallback="???",
        )
        return orig, dest

    def _parse_flight(self, raw_flight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        flight = raw_flight.get("flight") if isinstance(raw_flight.get("flight"), dict) else {}
        aircraft = raw_flight.get("aircraft") if isinstance(raw_flight.get("aircraft"), dict) else {}
        aircraft_model = aircraft.get("model") if isinstance(aircraft.get("model"), dict) else {}
        callsign = self._coerce_str(
            raw_flight.get("callsign"),
            raw_flight.get("flight"),
            flight.get("callsign"),
            flight.get("number"),
            raw_flight.get("identification"),
            fallback="NO CALL",
        )
        latitude = self._coerce_float(raw_flight.get("lat"), raw_flight.get("latitude"))
        longitude = self._coerce_float(raw_flight.get("lon"), raw_flight.get("longitude"))
        if latitude is None or longitude is None:
            return None

        distance_km = self._haversine_km(self.lat, self.lon, latitude, longitude)
        if distance_km > self.radius_km * 1.25:
            return None

        origin, destination = self._parse_route(raw_flight)
        aircraft_type = self._coerce_str(
            raw_flight.get("type"),
            raw_flight.get("aircraft_type"),
            raw_flight.get("equipment"),
            aircraft.get("type"),
            aircraft.get("code"),
            aircraft_model.get("code"),
            fallback="UNK",
        )

        return {
            "callsign": callsign,
            "origin": origin,
            "destination": destination,
            "distance_km": distance_km,
            "aircraft_type": aircraft_type,
        }

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        a_val = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        return radius * 2 * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))

    def _load_cached_flights(self, max_age: int) -> List[Dict[str, Any]]:
        cached = self.cache_manager.get(self.cache_key, max_age=max_age)
        if isinstance(cached, list):
            return [item for item in cached if isinstance(item, dict)]
        return []

    def _cache_flights(self, flights: List[Dict[str, Any]]) -> None:
        self.cache_manager.set(self.cache_key, flights, ttl=self.cache_seconds)

    def _handle_http_error(self, response: Response) -> None:
        if response.status_code == 401:
            self.status_code = "invalid_token"
            self.status_message = "Invalid token"
        elif response.status_code == 403:
            self.status_code = "endpoint_access"
            self.status_message = "Endpoint blocked"
        elif response.status_code == 429:
            self.status_code = "rate_limited"
            retry_after = response.headers.get("Retry-After")
            self.status_message = (
                f"Rate limited {retry_after}s" if retry_after else "Rate limited"
            )
        else:
            self.status_code = "api_error"
            self.status_message = f"HTTP {response.status_code}"

    def _fetch_live_flights(self) -> List[Dict[str, Any]]:
        last_error_response: Optional[Response] = None

        for endpoint_variant in self.endpoint_order:
            response = self.session.get(
                f"{self.api_base_url}/live/flight-positions/{endpoint_variant}",
                headers=self._request_headers(),
                params={
                    "bounds": self._build_bounds(),
                    "limit": self.result_limit,
                },
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                last_error_response = response
                # If the richer endpoint isn't available for this account, fall back
                # to light before surfacing the error to the UI.
                if endpoint_variant == "full" and response.status_code in {401, 403, 404}:
                    continue
                self._handle_http_error(response)
                return []

            payload = response.json()
            parsed_flights = []
            for raw_flight in self._extract_flights(payload):
                parsed = self._parse_flight(raw_flight)
                if parsed:
                    parsed_flights.append(parsed)

            parsed_flights.sort(key=lambda item: item["distance_km"])
            if parsed_flights or endpoint_variant == self.endpoint_order[-1]:
                return parsed_flights

        if last_error_response is not None:
            self._handle_http_error(last_error_response)
        return []

    def update(self) -> None:
        if not self.enabled:
            return

        self.api_token = self._resolve_api_token()
        current_time = time.time()
        if current_time - self.last_update < self.cache_seconds:
            return
        self.last_update = current_time

        cached_flights = self._load_cached_flights(self.cache_seconds)
        if cached_flights:
            self.current_flights = cached_flights
            self.current_flight = cached_flights[0]
            self.status_code = "ok"
            self.status_message = f"{len(cached_flights)} nearby"
            return

        if not self.api_token:
            self.current_flights = []
            self.current_flight = None
            self.status_code = "missing_token"
            self.status_message = "Add API token"
            return

        try:
            flights = self._fetch_live_flights()
        except requests.exceptions.RequestException as exc:
            self.logger.warning("FlightRadar24 request failed: %s", exc)
            flights = self._load_cached_flights(max(self.cache_seconds * 20, 600))
            self.current_flights = flights
            self.current_flight = flights[0] if flights else None
            self.status_code = "network_error"
            self.status_message = "Network error"
            return
        except ValueError as exc:
            self.logger.warning("FlightRadar24 response parsing failed: %s", exc)
            flights = self._load_cached_flights(max(self.cache_seconds * 20, 600))
            self.current_flights = flights
            self.current_flight = flights[0] if flights else None
            self.status_code = "api_error"
            self.status_message = "Bad response"
            return

        self.current_flights = flights
        self.current_flight = flights[0] if flights else None
        if flights:
            self._cache_flights(flights)
            self.status_code = "ok"
            self.status_message = f"{len(flights)} nearby"
        elif self.status_code in {"invalid_token", "endpoint_access", "rate_limited", "api_error"}:
            stale = self._load_cached_flights(max(self.cache_seconds * 20, 600))
            if stale:
                self.current_flights = stale
                self.current_flight = stale[0]
        else:
            self.status_code = "empty"
            self.status_message = "No flights"

    def _load_font(self, preferred_size: int = 6) -> ImageFont.ImageFont:
        display_font = getattr(self.display_manager, "extra_small_font", None)
        if display_font is not None:
            return display_font

        font_path = Path("assets/fonts/4x6-font.ttf")
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), preferred_size)
            except (OSError, ValueError):
                pass

        fallback_font = getattr(self.display_manager, "small_font", None)
        if fallback_font is not None:
            return fallback_font
        return ImageFont.load_default()

    def _draw_centered_lines(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        lines: List[Tuple[str, Tuple[int, int, int]]],
        font: ImageFont.ImageFont,
    ) -> None:
        line_height = self.display_manager.get_font_height(font)
        total_height = line_height * len(lines)
        y_pos = max(0, (height - total_height) // 2)
        for text, color in lines:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = max(0, (width - text_width) // 2)
            draw.text((x_pos, y_pos), text, font=font, fill=color)
            y_pos += line_height

    def _route_text(self, flight: Dict[str, Any]) -> str:
        origin = flight.get("origin", "???")
        destination = flight.get("destination", "???")
        return f"{origin}->{destination}"

    def _detail_text(self, flight: Dict[str, Any]) -> str:
        if self.show_aircraft_type:
            return flight.get("aircraft_type", "UNK")
        return f"{int(round(flight.get('distance_km', 0.0)))} km"

    def display(self, force_clear: bool = False) -> None:
        width = self.display_manager.width
        height = self.display_manager.height
        if force_clear:
            self.display_manager.clear()

        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = self._load_font()
        line_height = self.display_manager.get_font_height(font)
        can_show_three_lines = height >= line_height * 3

        if self.current_flight:
            lines: List[Tuple[str, Tuple[int, int, int]]] = [
                (self.current_flight["callsign"][:10], self.primary_color),
                (self._route_text(self.current_flight)[:14], self.secondary_color),
            ]
            if can_show_three_lines:
                lines.append((self._detail_text(self.current_flight)[:14], self.primary_color))
        else:
            status_line = self.status_message or "No data"
            lines = [("FR24", self.secondary_color), (status_line[:18], self.primary_color)]
            if can_show_three_lines and self.status_code == "missing_token":
                lines.append(("set token", self.primary_color))

        self._draw_centered_lines(draw, width, height, lines, font)
        self.display_manager.image = image
        self.display_manager.update_display()

    def validate_config(self) -> bool:
        if not super().validate_config():
            return False

        if not (-90.0 <= self.lat <= 90.0):
            self.logger.error("'lat' must be between -90 and 90")
            return False
        if not (-180.0 <= self.lon <= 180.0):
            self.logger.error("'lon' must be between -180 and 180")
            return False
        if self.radius_km <= 0:
            self.logger.error("'radius_km' must be positive")
            return False
        if self.cache_seconds <= 0:
            self.logger.error("'cache_seconds' must be positive")
            return False
        return True

    def on_config_change(self, new_config: Dict[str, Any]) -> None:
        super().on_config_change(new_config)
        self._apply_config(new_config)
        self.current_flights = []
        self.current_flight = None
        self.status_code = "config_changed"
        self.status_message = "Config updated"
        self.last_update = 0.0

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "status_code": self.status_code,
                "status_message": self.status_message,
                "flight_count": len(self.current_flights),
                "current_flight": self.current_flight,
            }
        )
        return info

    def cleanup(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.close()
        super().cleanup()
