"""PiAware local receiver flights plugin."""

from __future__ import annotations

import math
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common.logo_helper import LogoHelper
from src.plugin_system.base_plugin import BasePlugin


class PiAwareLocalFlightsPlugin(BasePlugin):
    """Display the nearest locally received aircraft from PiAware/dump1090."""

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
        self.logo_helper = LogoHelper(
            getattr(display_manager, "width", 64),
            getattr(display_manager, "height", 32),
            logger=self.logger,
        )
        self.current_aircraft: List[Dict[str, Any]] = []
        self.current_flight: Optional[Dict[str, Any]] = None
        self.status_code = "init"
        self.status_message = "Starting"
        self.last_update = 0.0

    def _apply_config(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.lat = float(self.config.get("lat", 0.0))
        self.lon = float(self.config.get("lon", 0.0))
        self.radius_km = float(self.config.get("radius_km", 120))
        self.cache_seconds = max(1, int(self.config.get("cache_seconds", 10)))
        self.display_duration = float(self.config.get("display_duration", 12))
        self.receiver_urls = self._normalize_urls(
            self.config.get(
                "receiver_urls",
                [
                    "/run/readsb/aircraft.json",
                    "http://127.0.0.1:8080/data/aircraft.json",
                    "http://127.0.0.1/skyaware/data/aircraft.json",
                ],
            )
        )
        self.request_timeout = max(1, int(self.config.get("request_timeout", 3)))
        self.max_seen_seconds = max(1, int(self.config.get("max_seen_seconds", 60)))
        self.show_altitude = bool(self.config.get("show_altitude", True))
        self.show_speed = bool(self.config.get("show_speed", True))
        self.show_airline_logo = bool(self.config.get("show_airline_logo", True))
        self.airline_logo_dir = Path(
            str(self.config.get("airline_logo_dir", "assets/airline_logos_piaware"))
        )
        self.airline_logo_fallback_dirs = [Path("assets/airline_logos_led"), Path("assets/airline_logos")]
        self.font_size = max(6, min(14, int(self.config.get("font_size", 9))))
        self.logo_max_width = max(12, min(64, int(self.config.get("logo_max_width", 40))))
        self.logo_text_gap = max(1, min(12, int(self.config.get("logo_text_gap", 1))))
        self.primary_color = self._normalize_color(self.config.get("primary_color"), (255, 255, 255))
        self.secondary_color = self._normalize_color(self.config.get("secondary_color"), (0, 255, 255))
        self.cache_key = f"{self.plugin_id}_{self.lat:.4f}_{self.lon:.4f}_{self.radius_km:.1f}"

    @staticmethod
    def _normalize_urls(value: Any) -> List[str]:
        if isinstance(value, list):
            urls = [str(item).strip() for item in value if str(item).strip()]
            if urls:
                return urls
        text = str(value or "").strip()
        return [text] if text else ["/run/readsb/aircraft.json"]

    @staticmethod
    def _normalize_color(value: Any, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return tuple(max(0, min(255, int(channel))) for channel in value)
            except (TypeError, ValueError):
                return fallback
        return fallback

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=1,
            connect=1,
            read=1,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

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

    @staticmethod
    def _normalize_airline_code(value: Any) -> str:
        if value is None:
            return ""
        text = "".join(char for char in str(value).upper().strip() if char.isalnum())
        return text[:3] if len(text) >= 2 else ""

    @staticmethod
    def _cardinal_from_bearing(bearing: Optional[float]) -> str:
        if bearing is None:
            return "?"
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((bearing + 22.5) // 45) % 8
        return directions[index]

    def _bearing_from_home(self, latitude: float, longitude: float) -> float:
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lon)
        lat2 = math.radians(latitude)
        lon2 = math.radians(longitude)
        dlon = lon2 - lon1
        x_val = math.sin(dlon) * math.cos(lat2)
        y_val = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x_val, y_val))
        return (bearing + 360.0) % 360.0

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

    def _fetch_receiver_payload(self) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None
        for source in self.receiver_urls:
            try:
                payload = self._read_receiver_source(source)
                if isinstance(payload, dict):
                    return payload
            except (requests.RequestException, ValueError) as exc:
                last_exception = exc
                continue
        if last_exception is not None:
            raise last_exception
        return {}

    def _read_receiver_source(self, source: str) -> Dict[str, Any]:
        if source.startswith("http://") or source.startswith("https://"):
            response = self.session.get(source, timeout=self.request_timeout)
            response.raise_for_status()
            return response.json()

        path_text = source[7:] if source.startswith("file://") else source
        path = Path(path_text)
        if not path.exists():
            raise requests.RequestException(f"receiver source not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _parse_aircraft(self, raw_aircraft: Dict[str, Any], now_ts: Optional[float]) -> Optional[Dict[str, Any]]:
        latitude = self._coerce_float(raw_aircraft.get("lat"))
        longitude = self._coerce_float(raw_aircraft.get("lon"))
        if latitude is None or longitude is None:
            return None

        seen_pos = self._coerce_float(raw_aircraft.get("seen_pos"), raw_aircraft.get("seen"))
        if seen_pos is not None and seen_pos > self.max_seen_seconds:
            return None
        if now_ts is not None and "seen" in raw_aircraft:
            seen_age = self._coerce_float(raw_aircraft.get("seen"))
            if seen_age is not None and seen_age > self.max_seen_seconds:
                return None

        distance_km = self._haversine_km(self.lat, self.lon, latitude, longitude)
        if distance_km > self.radius_km:
            return None

        callsign = self._coerce_str(raw_aircraft.get("flight"), raw_aircraft.get("hex"), fallback="NO HEX").upper()
        hex_code = self._coerce_str(raw_aircraft.get("hex")).upper()
        registration = self._coerce_str(raw_aircraft.get("r"), raw_aircraft.get("reg"))
        aircraft_type = self._coerce_str(raw_aircraft.get("t"), raw_aircraft.get("type"), fallback="UNK")
        altitude_ft = self._coerce_float(
            raw_aircraft.get("alt_baro"),
            raw_aircraft.get("alt_geom"),
            raw_aircraft.get("altitude"),
        )
        groundspeed = self._coerce_float(raw_aircraft.get("gs"), raw_aircraft.get("speed"))
        airline_code = self._normalize_airline_code(callsign[:3])
        bearing_home = self._bearing_from_home(latitude, longitude)

        return {
            "hex": hex_code,
            "callsign": callsign,
            "registration": registration,
            "aircraft_type": aircraft_type,
            "altitude_ft": altitude_ft,
            "groundspeed_kt": groundspeed,
            "distance_km": distance_km,
            "bearing_home": bearing_home,
            "direction_home": self._cardinal_from_bearing(bearing_home),
            "airline_code": airline_code,
        }

    def _extract_aircraft(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        now_ts = self._coerce_float(payload.get("now"))
        aircraft = payload.get("aircraft")
        if not isinstance(aircraft, list):
            return []
        parsed: List[Dict[str, Any]] = []
        for raw_aircraft in aircraft:
            if not isinstance(raw_aircraft, dict):
                continue
            item = self._parse_aircraft(raw_aircraft, now_ts)
            if item is not None:
                parsed.append(item)
        parsed.sort(key=lambda item: item["distance_km"])
        return parsed

    def update(self) -> None:
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self.last_update < self.cache_seconds:
            return
        self.last_update = current_time

        cached = self._load_cached_flights(self.cache_seconds)
        if cached:
            self.current_aircraft = cached
            self.current_flight = cached[0]
            self.status_code = "ok"
            self.status_message = f"{len(cached)} nearby"
            return

        try:
            payload = self._fetch_receiver_payload()
        except requests.RequestException as exc:
            self.logger.warning("PiAware receiver request failed: %s", exc)
            stale = self._load_cached_flights(max(self.cache_seconds * 12, 300))
            self.current_aircraft = stale
            self.current_flight = stale[0] if stale else None
            self.status_code = "receiver_error"
            self.status_message = "Receiver offline"
            return
        except ValueError as exc:
            self.logger.warning("PiAware receiver payload parse failed: %s", exc)
            self.current_aircraft = []
            self.current_flight = None
            self.status_code = "bad_payload"
            self.status_message = "Bad receiver data"
            return

        aircraft = self._extract_aircraft(payload)
        self.current_aircraft = aircraft
        self.current_flight = aircraft[0] if aircraft else None
        if aircraft:
            self._cache_flights(aircraft)
            self.status_code = "ok"
            self.status_message = f"{len(aircraft)} nearby"
        else:
            self.status_code = "empty"
            self.status_message = "No aircraft"

    def _load_font(self, preferred_size: Optional[int] = None) -> ImageFont.ImageFont:
        preferred_size = preferred_size or self.font_size
        font_path = Path("assets/fonts/4x6-font.ttf")
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), preferred_size)
            except (OSError, ValueError):
                pass
        fallback_font = getattr(self.display_manager, "small_font", None)
        if fallback_font is not None:
            return fallback_font
        display_font = getattr(self.display_manager, "extra_small_font", None)
        if display_font is not None:
            return display_font
        return ImageFont.load_default()

    def _load_airline_logo(self, flight: Optional[Dict[str, Any]], display_height: int) -> Optional[Image.Image]:
        if not self.show_airline_logo or not flight:
            return None
        airline_code = self._normalize_airline_code(flight.get("airline_code"))
        if not airline_code:
            return None
        target_width = min(self.logo_max_width, max(12, self.display_manager.width // 3))
        target_height = max(10, display_height - 2)
        logo_paths = [self.airline_logo_dir / f"{airline_code}.png"]
        logo_paths.extend(directory / f"{airline_code}.png" for directory in self.airline_logo_fallback_dirs)
        for logo_path in logo_paths:
            logo = self.logo_helper.load_logo(
                airline_code,
                logo_path,
                max_width=target_width,
                max_height=target_height,
            )
            if logo is not None:
                return logo
        return None

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        return text[:limit] if len(text) > limit else text

    def _route_like_text(self, flight: Dict[str, Any]) -> str:
        direction = flight.get("direction_home", "?")
        distance = int(round(flight.get("distance_km", 0.0)))
        return f"{direction} {distance}km"

    def _detail_text(self, flight: Dict[str, Any]) -> str:
        parts: List[str] = []
        if self.show_altitude and isinstance(flight.get("altitude_ft"), (int, float)):
            parts.append(f"{int(round(flight['altitude_ft']))}ft")
        if self.show_speed and isinstance(flight.get("groundspeed_kt"), (int, float)):
            parts.append(f"{int(round(flight['groundspeed_kt']))}kt")
        if parts:
            return " ".join(parts)
        return self._coerce_str(flight.get("registration"), flight.get("aircraft_type"), fallback="ADS-B")

    def _draw_lines(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        lines: List[Tuple[str, Tuple[int, int, int]]],
        font: ImageFont.ImageFont,
        x_offset: int = 0,
        left_bias: bool = False,
    ) -> None:
        line_height = self.display_manager.get_font_height(font)
        total_height = line_height * len(lines)
        y_pos = max(0, (height - total_height) // 2)
        usable_width = max(1, width - x_offset)
        block_width = 0
        if left_bias:
            for text, _color in lines:
                bbox = draw.textbbox((0, 0), text, font=font)
                block_width = max(block_width, bbox[2] - bbox[0])
        block_start = x_offset + (min(4, max(0, usable_width - block_width)) if left_bias and block_width else 0)
        for text, color in lines:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = block_start if left_bias else x_offset + max(0, (usable_width - text_width) // 2)
            draw.text((x_pos, y_pos), text, font=font, fill=color)
            y_pos += line_height

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
        airline_logo = self._load_airline_logo(self.current_flight, height)
        text_x_offset = 0

        if airline_logo is not None:
            logo_x = 2
            logo_y = max(0, (height - airline_logo.height) // 2)
            image.paste(airline_logo, (logo_x, logo_y), airline_logo)
            text_x_offset = min(width - 1, logo_x + airline_logo.width + self.logo_text_gap)

        if self.current_flight:
            lines: List[Tuple[str, Tuple[int, int, int]]] = [
                (self._truncate(self.current_flight["callsign"], 10), self.primary_color),
                (self._truncate(self._route_like_text(self.current_flight), 14), self.secondary_color),
            ]
            if can_show_three_lines:
                lines.append((self._truncate(self._detail_text(self.current_flight), 14), self.primary_color))
        else:
            lines = [("ADS-B", self.secondary_color), (self._truncate(self.status_message or "No data", 18), self.primary_color)]

        self._draw_lines(
            draw,
            width,
            height,
            lines,
            font,
            x_offset=text_x_offset,
            left_bias=airline_logo is not None,
        )
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
        return True

    def on_config_change(self, new_config: Dict[str, Any]) -> None:
        super().on_config_change(new_config)
        self._apply_config(new_config)
        self.current_aircraft = []
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
                "aircraft_count": len(self.current_aircraft),
                "current_flight": self.current_flight,
            }
        )
        return info

    def cleanup(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.close()
        super().cleanup()
