"""
HealDeal — Geocoding Utility
Converts place names → (lat, lon) using Nominatim with local caching.
"""
import json, time
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

CACHE_PATH = Path(__file__).parent.parent / "data" / "geocode_cache.json"
_geolocator = None
_geocode_once = None


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _get_geolocator():
    global _geolocator
    if _geolocator is None:
        _geolocator = Nominatim(user_agent="healdeal-demo/1.0 (educational project)", timeout=10)
    return _geolocator


def geocode(place: str) -> dict | None:
    """
    Geocode a place name. Returns dict with lat, lon, display_name or None.
    Results are cached locally to avoid repeated API calls.
    """
    if not place or not str(place).strip():
        return None

    key = str(place).strip().lower()
    cache = _load_cache()

    if key in cache:
        return cache[key]

    try:
        global _geocode_once
        geo = _get_geolocator()
        # Reuse one limiter across requests to respect the public service's rate guidance.
        if _geocode_once is None:
            _geocode_once = RateLimiter(geo.geocode, min_delay_seconds=1.0, swallow_exceptions=False)
        loc = _geocode_once(place, exactly_one=True)
        if loc:
            result = {
                "lat": loc.latitude,
                "lon": loc.longitude,
                "display_name": loc.address,
            }
            cache[key] = result
            _save_cache(cache)
            return result
    except (GeocoderTimedOut, GeocoderServiceError):
        # Try without cache
        return None
    except Exception:
        return None
    return None


# Quick lookup table for major Indian cities (avoids API calls in demo)
CITY_COORDS = {
    "delhi":         {"lat": 28.6139, "lon": 77.2090, "display_name": "New Delhi, India"},
    "new delhi":     {"lat": 28.6139, "lon": 77.2090, "display_name": "New Delhi, India"},
    "mumbai":        {"lat": 19.0760, "lon": 72.8777, "display_name": "Mumbai, Maharashtra, India"},
    "bengaluru":     {"lat": 12.9716, "lon": 77.5946, "display_name": "Bengaluru, Karnataka, India"},
    "bangalore":     {"lat": 12.9716, "lon": 77.5946, "display_name": "Bengaluru, Karnataka, India"},
    "hyderabad":     {"lat": 17.3850, "lon": 78.4867, "display_name": "Hyderabad, Telangana, India"},
    "chennai":       {"lat": 13.0827, "lon": 80.2707, "display_name": "Chennai, Tamil Nadu, India"},
    "kolkata":       {"lat": 22.5726, "lon": 88.3639, "display_name": "Kolkata, West Bengal, India"},
    "pune":          {"lat": 18.5204, "lon": 73.8567, "display_name": "Pune, Maharashtra, India"},
    "ahmedabad":     {"lat": 23.0225, "lon": 72.5714, "display_name": "Ahmedabad, Gujarat, India"},
    "jaipur":        {"lat": 26.9124, "lon": 75.7873, "display_name": "Jaipur, Rajasthan, India"},
    "lucknow":       {"lat": 26.8467, "lon": 80.9462, "display_name": "Lucknow, UP, India"},
    "chandigarh":    {"lat": 30.7333, "lon": 76.7794, "display_name": "Chandigarh, India"},
    "kochi":         {"lat": 9.9312,  "lon": 76.2673, "display_name": "Kochi, Kerala, India"},
    "bhopal":        {"lat": 23.2599, "lon": 77.4126, "display_name": "Bhopal, MP, India"},
    "nagpur":        {"lat": 21.1458, "lon": 79.0882, "display_name": "Nagpur, Maharashtra, India"},
    "indore":        {"lat": 22.7196, "lon": 75.8577, "display_name": "Indore, MP, India"},
    "surat":         {"lat": 21.1702, "lon": 72.8311, "display_name": "Surat, Gujarat, India"},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "display_name": "Visakhapatnam, AP, India"},
    "patna":         {"lat": 25.5941, "lon": 85.1376, "display_name": "Patna, Bihar, India"},
    "mysuru":        {"lat": 12.2958, "lon": 76.6394, "display_name": "Mysuru, Karnataka, India"},
    "mysore":        {"lat": 12.2958, "lon": 76.6394, "display_name": "Mysuru, Karnataka, India"},
    "coimbatore":    {"lat": 11.0168, "lon": 76.9558, "display_name": "Coimbatore, TN, India"},
}


def geocode_smart(place: str) -> dict | None:
    """First checks local city table, then falls back to Nominatim."""
    if not isinstance(place, str) or not place.strip():
        return None
    key = place.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    return geocode(place)
