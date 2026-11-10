"""
HealDeal — User Authentication & Session Management
Simple file-based auth (no external DB needed for capstone demo).
"""
import json
import time
import os
import re
import tempfile
from typing import Optional
from pathlib import Path
from werkzeug.security import check_password_hash, generate_password_hash

USERS_PATH = Path(__file__).parent.parent / "data" / "users.json"


def _hash(password: str) -> str:
    """Use a strong adaptive hash for new passwords."""
    return generate_password_hash(password)


def _load() -> dict:
    if not USERS_PATH.exists():
        return {}
    try:
        data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Account store must contain a JSON object.")
        return data
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Account store is not valid JSON: {USERS_PATH}") from exc


def _save(data: dict):
    """Atomically persist demo accounts; this file store is for local demos only."""
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix="users-", suffix=".tmp", dir=USERS_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, USERS_PATH)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _find_username_casefold(data: dict, username: str) -> Optional[str]:
    """Return canonical stored username matching input case-insensitively."""
    target = username.strip().lower()
    for key in data.keys():
        if key.lower() == target:
            return key
    return None


def register(username: str, password: str, email: str = "") -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    username = str(username or "").strip()
    password = str(password or "")
    email = str(email or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        return False, "Username must be 3–32 characters using letters, numbers, dot, underscore or hyphen."
    if len(password) < 8 or len(password) > 128:
        return False, "Password must be between 8 and 128 characters."
    if email and (len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)):
        return False, "Enter a valid email address."

    data = _load()
    if _find_username_casefold(data, username):
        return False, "Username already exists."

    data[username] = {
        "password_hash": _hash(password),
        "email": email,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "xp": 0,
        "level": 1,
        "favorites": [],
        "search_history": [],
        "achievements": [],
    }
    _save(data)
    return True, "Registration successful!"


def login(username: str, password: str) -> tuple[bool, str, dict]:
    """Login a user. Returns (success, message, user_data)."""
    data = _load()
    canonical = _find_username_casefold(data, str(username or "").strip())
    if not canonical:
        return False, "User not found.", {}

    user = data[canonical]
    stored_hash = user.get("password_hash", "")

    # Backward compatibility for older SHA256-only hashes.
    if stored_hash and not stored_hash.startswith(("scrypt:", "pbkdf2:")):
        import hashlib
        if hashlib.sha256(password.encode()).hexdigest() != stored_hash:
            return False, "Incorrect password.", {}
        user["password_hash"] = _hash(password)
        data[canonical] = user
        _save(data)
    elif not check_password_hash(stored_hash, password):
        return False, "Incorrect password.", {}

    safe_user = {**user, "username": canonical}
    return True, "Login successful!", safe_user


def get_user(username: str) -> dict:
    data = _load()
    canonical = _find_username_casefold(data, username)
    return data.get(canonical, {}) if canonical else {}


def update_user(username: str, updates: dict):
    data = _load()
    canonical = _find_username_casefold(data, username)
    if canonical:
        data[canonical].update(updates)
        _save(data)


def add_xp(username: str, amount: int) -> dict:
    data = _load()
    canonical = _find_username_casefold(data, username)
    if not canonical:
        return {}
    data[canonical]["xp"] = data[canonical].get("xp", 0) + amount
    data[canonical]["level"] = 1 + data[canonical]["xp"] // 100
    # check achievements
    achievements = data[canonical].get("achievements", [])
    xp = data[canonical]["xp"]
    if xp >= 100 and "First 100 XP" not in achievements:
        achievements.append("First 100 XP 🏅")
    if xp >= 500 and "Healthcare Explorer 🌟" not in achievements:
        achievements.append("Healthcare Explorer 🌟")
    if len(data[canonical].get("favorites", [])) >= 5 and "Collector 📋" not in achievements:
        achievements.append("Collector 📋")
    data[canonical]["achievements"] = achievements
    _save(data)
    return data[canonical]


def save_favorite(username: str, hospital: dict) -> tuple[bool, str]:
    data = _load()
    canonical = _find_username_casefold(data, username)
    if not canonical:
        return False, "User not found."
    favs = data[canonical].get("favorites", [])
    if any(f.get("hospital_id") == hospital.get("hospital_id") for f in favs):
        return False, "Already in favorites."
    favs.append(hospital)
    data[canonical]["favorites"] = favs
    _save(data)
    return True, "Added to favorites!"


def remove_favorite(username: str, hospital_id: str):
    data = _load()
    canonical = _find_username_casefold(data, username)
    if canonical:
        data[canonical]["favorites"] = [
            f for f in data[canonical].get("favorites", [])
            if f.get("hospital_id") != hospital_id
        ]
        _save(data)


def append_search(username: str, entry: dict):
    data = _load()
    canonical = _find_username_casefold(data, username)
    if canonical:
        hist = data[canonical].get("search_history", [])
        hist.insert(0, {**entry, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        data[canonical]["search_history"] = hist[:50]   # keep last 50
        _save(data)
