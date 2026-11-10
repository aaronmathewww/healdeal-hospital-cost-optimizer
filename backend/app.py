"""
HealDeal v2.0 — Flask REST API Backend
B.Tech Capstone Project

Run:
    pip install -r requirements.txt
    python backend/app.py

API Base: http://localhost:5000/api
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys, secrets, time, math, re
from pathlib import Path
import requests

# ── Path setup ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.ces_engine import CESEngine, haversine_km
from ml.sentiment  import analyze_text, extract_keywords
from utils.geocoder import geocode_smart
from utils.auth import (register, login, get_user, add_xp,
                         save_favorite, remove_favorite, append_search)
from backend.config import Settings

# ── App Init ────────────────────────────────────────────
app = Flask(__name__,
            static_folder=str(ROOT / "frontend"),
            static_url_path="")
app.config["SECRET_KEY"] = Settings.SECRET_KEY

if Settings.CORS_ORIGINS.strip() == "*":
    cors_origins = "*"
else:
    cors_origins = [o.strip() for o in Settings.CORS_ORIGINS.split(",") if o.strip()]

CORS(app, resources={r"/api/*": {"origins": cors_origins}})

# ── In-memory token store {token: username} ─────────────
_TOKENS: dict[str, tuple[str, float]] = {}
TOKEN_TTL_SECONDS = 8 * 60 * 60


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────
def _require_auth():
    """Extract token from header and return username or None."""
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    session = _TOKENS.get(token.strip())
    if not session:
        return None
    username, expires_at = session
    if expires_at <= time.time():
        _TOKENS.pop(token.strip(), None)
        return None
    return username


def _ok(data=None, **kwargs):
    return jsonify({"ok": True, "data": data if data is not None else {}, **kwargs})


def _err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


@app.before_request
def _load_data_once():
    if not hasattr(app, "_df_loaded"):
        _load_df()
        app._df_loaded = True


def _load_df():
    path = ROOT / "data" / "hospitals.csv"
    if not path.exists():
        app._df = pd.DataFrame()
        return
    df = pd.read_csv(path)
    for c in ["avg_rating", "avg_cost_for_procedure", "latitude",
              "longitude", "bed_count", "num_reviews", "sentiment_score"]:
        if c not in df.columns:
            df[c] = 0
    app._df = df


def _df():
    return getattr(app, "_df", pd.DataFrame())


# ─────────────────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def api_register():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("A JSON object is required.")
    username = str(body.get("username", ""))[:64]
    password = str(body.get("password", ""))
    email = str(body.get("email", ""))[:254]
    if email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return _err("Enter a valid email address.")
    ok, msg = register(username, password, email)
    return _ok({"message": msg}) if ok else _err(msg)


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("A JSON object is required.")
    ok, msg, udata = login(str(body.get("username", ""))[:64], str(body.get("password", "")))
    if not ok:
        return _err(msg, 401)
    token = secrets.token_urlsafe(32)
    canonical_username = udata.get("username", body.get("username", ""))
    _TOKENS[token] = (canonical_username, time.time() + TOKEN_TTL_SECONDS)
    udata_safe = {k: v for k, v in udata.items() if k != "password_hash"}
    return _ok({"token": token, "expires_in": TOKEN_TTL_SECONDS, "user": udata_safe})


@app.route("/api/auth/profile", methods=["GET"])
def api_profile():
    username = _require_auth()
    if not username:
        return _err("Unauthorized", 401)
    udata = get_user(username)
    udata_safe = {k: v for k, v in udata.items() if k != "password_hash"}
    udata_safe["username"] = username
    return _ok(udata_safe)


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer":
        _TOKENS.pop(token.strip(), None)
    return _ok({"message": "Logged out."})


# ─────────────────────────────────────────────────────────
#  HOSPITAL ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/api/hospitals/search", methods=["POST"])
def api_search():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("A JSON object is required.")

    location = str(body.get("location", "")).strip()
    if not location or len(location) > 120:
        return _err("Enter a location of 1–120 characters.")

    df = _df().copy()
    if df.empty:
        return _err("Demo dataset is unavailable.", 503)

    specialty = str(body.get("specialty", "Any"))
    tier = str(body.get("tier", "Any"))
    insurance = str(body.get("insurance", "Any"))
    if specialty != "Any" and specialty not in set(df["primary_specialty"].dropna()):
        return _err("Unknown specialty filter.")
    if tier != "Any" and tier not in {"Premium", "Mid-Range", "Budget"}:
        return _err("Unknown tier filter.")
    if insurance != "Any" and len(insurance) > 80:
        return _err("Insurance filter is too long.")

    try:
        max_km = float(body.get("max_km", 50))
        page = int(body.get("page", 1))
        per_page = int(body.get("per_page", 15))
    except (TypeError, ValueError, OverflowError):
        return _err("Distance and pagination values must be numeric.")
    if not math.isfinite(max_km) or not 1 <= max_km <= 500:
        return _err("Maximum distance must be between 1 and 500 km.")
    if page < 1 or not 1 <= per_page <= 50:
        return _err("Page must be positive and page size must be between 1 and 50.")

    geo = geocode_smart(location)
    if geo is None:
        return _err("Could not resolve location. Try a major city name.")
    user_lat, user_lon = float(geo["lat"]), float(geo["lon"])

    if specialty != "Any":
        df = df[df["primary_specialty"] == specialty]
    if tier != "Any":
        df = df[df["tier"] == tier]
    accredited_only = body.get("accredited_only", False) is True
    if accredited_only:
        df = df[df["accredited"].astype(str).str.lower() == "true"]
    if insurance != "Any":
        df = df[df["insurance_accepted"].fillna("").str.contains(insurance, regex=False, case=False)]

    df["distance_km"] = df.apply(
        lambda row: haversine_km(user_lat, user_lon, row["latitude"], row["longitude"]), axis=1
    )
    df = df[np.isfinite(df["distance_km"]) & (df["distance_km"] <= max_km)].copy()
    if df.empty:
        return _ok({"hospitals": [], "total": 0, "pages": 0, "page": page,
                    "location": geo, "summary": None,
                    "data_notice": "All records are synthetic demonstration data."})

    use_ml = body.get("use_ml", False) is True
    predicted_costs = None
    cost_source = "synthetic_dataset_cost"
    if use_ml:
        try:
            from ml.cost_model import predict_cost
            predicted_costs = predict_cost(df)
            cost_source = "experimental_model_on_synthetic_data"
        except Exception:
            predicted_costs = None

    defaults = {"cost": 0.40, "rating": 0.30, "distance": 0.20, "sentiment": 0.10}
    weights = body.get("weights", defaults)
    if not isinstance(weights, dict):
        weights = defaults
    clean_weights = {}
    for key in defaults:
        try:
            value = float(weights.get(key, defaults[key]))
        except (TypeError, ValueError, OverflowError):
            value = defaults[key]
        clean_weights[key] = value if math.isfinite(value) and value >= 0 else defaults[key]
    if sum(clean_weights.values()) <= 0:
        clean_weights = defaults

    engine = CESEngine(clean_weights)
    scored = engine.score(df, user_lat, user_lon, predicted_costs)
    total = len(scored)
    start = (page - 1) * per_page
    page_df = scored.iloc[start:start + per_page].copy()
    cols = ["hospital_id", "name", "city", "primary_specialty", "tier",
            "pred_cost", "avg_cost_for_procedure", "avg_rating", "distance_km",
            "sentiment_score", "bed_count", "accredited", "insurance_accepted",
            "primary_procedure", "latitude", "longitude", "CES", "percentile_rank",
            "cost_n", "rating_n", "dist_n", "sent_n"]
    cols = [column for column in cols if column in page_df.columns]
    records = page_df[cols].replace({np.nan: None}).to_dict(orient="records")

    username = _require_auth()
    if username:
        append_search(username, {"location": location, "specialty": specialty,
                                 "results": total, "top": records[0]["name"] if records else ""})
        add_xp(username, 10)

    return _ok({
        "hospitals": records, "total": total,
        "pages": int(math.ceil(total / per_page)), "page": page,
        "location": geo, "cost_source": cost_source,
        "data_notice": "Synthetic demonstration records only; not real hospitals, quotations, ratings, reviews, accreditation, insurance acceptance or facility locations.",
        "summary": {
            "min_cost": float(scored["pred_cost"].min()),
            "avg_cost": float(scored["pred_cost"].mean()),
            "max_rating": float(scored["avg_rating"].max()),
            "nearest_km": float(scored["distance_km"].min()),
        },
    })


@app.route("/api/hospitals/<hospital_id>", methods=["GET"])
def api_hospital_detail(hospital_id):
    df = _df()
    row = df[df["hospital_id"] == hospital_id]
    if row.empty:
        return _err("Hospital not found", 404)
    rec = row.iloc[0].replace({np.nan: None}).to_dict()
    # Sentiment per review
    if "reviews" in rec and rec["reviews"]:
        parts = str(rec["reviews"]).split("|")
        rec["review_details"] = [
            {"text": r.strip(), **analyze_text(r)} for r in parts if r.strip()
        ]
        rec["keywords"] = extract_keywords(rec["reviews"])
    return _ok(rec)


@app.route("/api/hospitals/meta", methods=["GET"])
def api_meta():
    df = _df()
    return _ok({
        "total": len(df),
        "cities": sorted(df["city"].dropna().unique().tolist()) if not df.empty else [],
        "specialties": sorted(df["primary_specialty"].dropna().unique().tolist()) if not df.empty else [],
        "tiers": ["Any", "Premium", "Mid-Range", "Budget"],
        "insurances": ["Any", "Sample Insurance A", "Sample Insurance B",
                       "Sample Insurance C", "Sample Insurance D", "Sample Insurance E"],
    })


# ─────────────────────────────────────────────────────────
#  FAVORITES ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/api/user/favorites", methods=["GET"])
def api_get_favorites():
    username = _require_auth()
    if not username:
        return _err("Unauthorized", 401)
    udata = get_user(username)
    return _ok(udata.get("favorites", []))


@app.route("/api/user/favorites", methods=["POST"])
def api_add_favorite():
    username = _require_auth()
    if not username:
        return _err("Unauthorized", 401)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("A JSON object is required.")
    hospital_id = str(body.get("hospital_id", ""))[:32]
    df = _df()
    match = df[df["hospital_id"] == hospital_id]
    if match.empty:
        return _err("Unknown demo record.", 404)
    row = match.iloc[0]
    # Rebuild the saved record from trusted dataset fields; never trust client-supplied names/prices.
    hospital = {
        "hospital_id": str(row["hospital_id"]),
        "name": str(row["name"]),
        "city": str(row["city"]),
        "pred_cost": float(row["avg_cost_for_procedure"]),
        "avg_rating": float(row["avg_rating"]),
        "specialty": str(row["primary_specialty"]),
    }
    ok, msg = save_favorite(username, hospital)
    if ok:
        add_xp(username, 30)
    return _ok({"message": msg}) if ok else _err(msg)


@app.route("/api/user/favorites/<hospital_id>", methods=["DELETE"])
def api_remove_favorite(hospital_id):
    username = _require_auth()
    if not username:
        return _err("Unauthorized", 401)
    remove_favorite(username, hospital_id)
    return _ok({"message": "Removed."})


# ─────────────────────────────────────────────────────────
#  HISTORY & ANALYTICS
# ─────────────────────────────────────────────────────────
@app.route("/api/user/history", methods=["GET"])
def api_history():
    username = _require_auth()
    if not username:
        return _err("Unauthorized", 401)
    udata = get_user(username)
    return _ok(udata.get("search_history", []))


@app.route("/api/analytics/overview", methods=["GET"])
def api_analytics():
    df = _df()
    if df.empty:
        return _ok({})
    by_city = df.groupby("city").agg(
        count=("hospital_id", "count"),
        avg_cost=("avg_cost_for_procedure", "mean"),
        avg_rating=("avg_rating", "mean"),
    ).reset_index().to_dict(orient="records")
    by_specialty = df.groupby("primary_specialty").agg(
        count=("hospital_id", "count"),
        avg_cost=("avg_cost_for_procedure", "mean"),
    ).reset_index().sort_values("count", ascending=False).head(10).to_dict(orient="records")
    return _ok({
        "total_hospitals": len(df),
        "total_cities":    df["city"].nunique(),
        "total_specialties": df["primary_specialty"].nunique(),
        "avg_rating":      round(float(df["avg_rating"].mean()), 2),
        "avg_cost":        round(float(df["avg_cost_for_procedure"].mean()), 0),
        "by_city":         by_city,
        "by_specialty":    by_specialty,
        "tier_dist":       df["tier"].value_counts().to_dict(),
        "sentiment_dist":  {
            "positive": int((df["sentiment_score"] > 0.6).sum()),
            "neutral":  int(((df["sentiment_score"] >= 0.45) & (df["sentiment_score"] <= 0.6)).sum()),
            "negative": int((df["sentiment_score"] < 0.45).sum()),
        }
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    return _ok({
        "service": "healdeal-backend",
        "status": "healthy",
        "dataset_loaded": not _df().empty,
        "chat_provider": "gemini" if Settings.GEMINI_API_KEY else ("huggingface" if Settings.HF_API_KEY else "offline"),
        "dataset_type": "synthetic_demo_data",
    })


def _normalize_model_name(model_name: str) -> str:
    model_name = str(model_name or "").strip()
    if model_name.startswith("models/"):
        return model_name.split("models/", 1)[1]
    return model_name


def _resolve_gemini_model(api_key: str, requested_model: str) -> str:
    """Resolve a valid Gemini model that supports generateContent.

    Uses ListModels to avoid hard failures when model aliases are retired.
    """
    requested = _normalize_model_name(requested_model) or "gemini-2.0-flash"
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

    try:
        resp = requests.get(list_url, timeout=Settings.CHAT_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            return requested

        models = resp.json().get("models", [])
        available = []
        for model in models:
            methods = model.get("supportedGenerationMethods", []) or []
            if "generateContent" not in methods:
                continue
            normalized = _normalize_model_name(model.get("name", ""))
            if normalized:
                available.append(normalized)

        if not available:
            return requested

        if requested in available:
            return requested

        preferred = [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        for name in preferred:
            if name in available:
                return name

        flash_models = sorted([m for m in available if "flash" in m])
        if flash_models:
            return flash_models[0]

        return sorted(available)[0]
    except Exception:
        return requested


def _format_gemini_error(resp: requests.Response, model_name: str) -> str:
    """Return concise user-facing Gemini API error details."""
    try:
        payload = resp.json()
        err = payload.get("error", {})
        status = err.get("status", "UNKNOWN")
        message = err.get("message", "Gemini request failed.")

        if resp.status_code == 429 or status == "RESOURCE_EXHAUSTED":
            retry_hint = ""
            for item in err.get("details", []):
                if item.get("@type", "").endswith("RetryInfo"):
                    delay = item.get("retryDelay", "")
                    if delay:
                        retry_hint = f" Retry after about {delay}."
                    break
            return (
                f"Gemini quota exceeded for model '{model_name}'."
                " Check your API quota/billing in Google AI Studio."
                f"{retry_hint}"
            )

        return f"Gemini API error ({status}, model: {model_name}): {message}"
    except Exception:
        return f"Gemini API error ({resp.status_code}, model: {model_name})."


def _offline_healbot_reply(user_text: str) -> str:
    """Safe, non-diagnostic offline responses when external AI is unavailable."""
    q = (user_text or "").strip().lower()
    if not q:
        return "I can explain the demo's cost-effectiveness score, comparison filters, and synthetic dataset limitations. What would you like to understand?"
    if "ces" in q or "cost-effectiveness" in q:
        return "HealDeal's demo CES combines normalized sample cost, distance, rating, and review sentiment using user-selected weights. A lower score ranks higher within the current synthetic result set; it is not a measure of clinical quality."
    if any(term in q for term in ("cost", "price", "bill", "insurance", "coverage")):
        return "The bundled prices and insurance tags are synthetic demonstration values, not real quotations or coverage information. Contact the hospital and insurer directly for current, case-specific costs and eligibility."
    if any(term in q for term in ("symptom", "pain", "diagnose", "medicine", "treatment")):
        return "I can't diagnose or recommend treatment. Please discuss symptoms and care options with a qualified healthcare professional; seek urgent medical attention for severe or rapidly worsening symptoms."
    return "The AI service is unavailable right now. I can still explain the demo interface and its scoring logic, but please verify real-world healthcare information directly with qualified providers."


def _messages_to_hf_chat_messages(system_prompt: str, messages: list[dict], max_turns: int = 10) -> list[dict]:
    """Convert chat history to OpenAI-compatible chat message format."""
    trimmed = messages[-max_turns:]
    chat_messages = [{"role": "system", "content": system_prompt.strip()}]
    for msg in trimmed:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        text = str(msg.get("content", "")).strip()
        if text:
            chat_messages.append({"role": role, "content": text})
    return chat_messages


def _try_huggingface_reply(system_prompt: str, messages: list[dict]) -> tuple[str | None, str | None]:
    """Try Hugging Face Inference API as fallback provider."""
    if not Settings.HF_API_KEY:
        return None, "HF_API_KEY not configured."

    chat_messages = _messages_to_hf_chat_messages(system_prompt, messages)
    endpoint = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {Settings.HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": Settings.HF_MODEL,
        "messages": chat_messages,
        "max_tokens": 320,
        "temperature": 0.4,
    }

    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=Settings.CHAT_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 400:
            return None, f"Hugging Face API error ({resp.status_code}): {resp.text}"

        data = resp.json()
        text = ""
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices:
                text = str(choices[0].get("message", {}).get("content", "")).strip()

        if not text:
            return None, "Hugging Face returned empty output."

        return text, None
    except Exception as ex:
        return None, f"Hugging Face request failed: {ex}"


# ─────────────────────────────────────────────────────────
#  AI CHATBOT ROUTE
# ─────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("A JSON object is required.")
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages or len(messages) > 20:
        return _err("Provide between 1 and 20 chat messages.")
    if any(
        not isinstance(item, dict)
        or item.get("role") not in {"user", "assistant"}
        or not isinstance(item.get("content"), str)
        or len(item.get("content", "")) > 4000
        for item in messages
    ):
        return _err("Each chat message must use a user/assistant role and be no longer than 4,000 characters.")
    if not any(item.get("role") == "user" for item in messages):
        return _err("At least one user message is required.")
    api_key  = Settings.GEMINI_API_KEY

    system_prompt = """You are HealBot, an expert AI healthcare assistant embedded in HealDeal — 
an AI-powered hospital cost optimizer for India.

You are an informational assistant, not a clinician, insurer, hospital representative, or emergency service.
You may explain the application's CES formula, help users prepare questions for providers, and explain general healthcare terminology.
The bundled hospital records, prices, ratings, reviews, insurance tags, accreditation flags and coordinates are synthetic demonstration data. Never describe them as real or verified.
Do not invent current hospital prices, provider details, coverage decisions, diagnoses, treatment plans or outcomes. For actual quotations and coverage, direct users to the hospital and insurer.
Do not diagnose, prescribe, or recommend delaying care. For urgent symptoms, advise contacting local emergency services or attending the nearest emergency department.
Use clear, empathetic, concise language. State uncertainty plainly."""

    # Gemini expects "role" + "parts" message format.
    contents = []
    latest_user_text = ""
    for item in messages:
        role = item.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        text = str(item.get("content", "")).strip()
        if text:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
            if role == "user":
                latest_user_text = text

    if not contents:
        return _err("At least one chat message is required.", 400)

    if not api_key:
        hf_reply, hf_reason = _try_huggingface_reply(system_prompt, messages)
        if hf_reply:
            return _ok({
                "reply": hf_reply,
                "model": Settings.HF_MODEL,
                "provider": "huggingface",
                "fallback": True,
                "reason": "Gemini API key missing; used Hugging Face fallback.",
            })
        return _ok({
            "reply": _offline_healbot_reply(latest_user_text),
            "model": "fallback-offline",
            "fallback": True,
            "reason": f"Gemini API key missing. {hf_reason or ''}".strip(),
        })

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 700,
        },
    }

    requested_model = Settings.GEMINI_MODEL
    resolved_model = _resolve_gemini_model(api_key, requested_model)

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{resolved_model}:generateContent?key={api_key}"
    )

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=Settings.CHAT_TIMEOUT_SECONDS,
        )
        if resp.status_code == 404:
            # Retry once using fresh auto-resolution in case configured model is stale.
            fallback_model = _resolve_gemini_model(api_key, "")
            if fallback_model != resolved_model:
                endpoint = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{fallback_model}:generateContent?key={api_key}"
                )
                resp = requests.post(
                    endpoint,
                    json=payload,
                    timeout=Settings.CHAT_TIMEOUT_SECONDS,
                )
                resolved_model = fallback_model

        if resp.status_code >= 400:
            hf_reply, hf_reason = _try_huggingface_reply(system_prompt, messages)
            if hf_reply:
                return _ok({
                    "reply": hf_reply,
                    "model": Settings.HF_MODEL,
                    "provider": "huggingface",
                    "fallback": True,
                    "reason": _format_gemini_error(resp, resolved_model),
                })
            return _ok({
                "reply": _offline_healbot_reply(latest_user_text),
                "model": "fallback-offline",
                "fallback": True,
                "reason": f"{_format_gemini_error(resp, resolved_model)} | {hf_reason or 'HF unavailable.'}",
            })

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            hf_reply, hf_reason = _try_huggingface_reply(system_prompt, messages)
            if hf_reply:
                return _ok({
                    "reply": hf_reply,
                    "model": Settings.HF_MODEL,
                    "provider": "huggingface",
                    "fallback": True,
                    "reason": "Gemini returned no response candidates.",
                })
            return _ok({
                "reply": _offline_healbot_reply(latest_user_text),
                "model": "fallback-offline",
                "fallback": True,
                "reason": f"Gemini returned no response candidates. {hf_reason or ''}".strip(),
            })

        parts = candidates[0].get("content", {}).get("parts", [])
        reply = "\n".join(p.get("text", "") for p in parts).strip()
        if not reply:
            hf_reply, hf_reason = _try_huggingface_reply(system_prompt, messages)
            if hf_reply:
                return _ok({
                    "reply": hf_reply,
                    "model": Settings.HF_MODEL,
                    "provider": "huggingface",
                    "fallback": True,
                    "reason": "Gemini returned an empty response.",
                })
            return _ok({
                "reply": _offline_healbot_reply(latest_user_text),
                "model": "fallback-offline",
                "fallback": True,
                "reason": f"Gemini returned an empty response. {hf_reason or ''}".strip(),
            })
        return _ok({"reply": reply, "model": resolved_model})
    except Exception as e:
        hf_reply, hf_reason = _try_huggingface_reply(system_prompt, messages)
        if hf_reply:
            return _ok({
                "reply": hf_reply,
                "model": Settings.HF_MODEL,
                "provider": "huggingface",
                "fallback": True,
                "reason": str(e),
            })
        return _ok({
            "reply": _offline_healbot_reply(latest_user_text),
            "model": "fallback-offline",
            "fallback": True,
            "reason": f"{e} | {hf_reason or 'HF unavailable.'}",
        })


@app.errorhandler(404)
def handle_not_found(error):
    if request.path.startswith("/api/"):
        return _err("API endpoint not found.", 404)
    return error


@app.errorhandler(500)
def handle_server_error(error):
    app.logger.exception("Unhandled application error")
    if request.path.startswith("/api/"):
        return _err("An internal server error occurred.", 500)
    return "Internal server error", 500


# ─────────────────────────────────────────────────────────
#  Serve Frontend
# ─────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and (ROOT / "frontend" / path).exists():
        return send_from_directory(str(ROOT / "frontend"), path)
    return send_from_directory(str(ROOT / "frontend"), "index.html")


# ─────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏥 HealDeal Backend starting...")
    print(f"   Dataset: {len(_df())} hospitals" if not _df().empty else "   ⚠️  Run data/generate_dataset.py first")
    print(f"   URL: http://localhost:{Settings.PORT}")
    print(f"   Chat model: {Settings.GEMINI_MODEL}")
    app.run(debug=Settings.DEBUG, port=Settings.PORT, host=Settings.HOST)
