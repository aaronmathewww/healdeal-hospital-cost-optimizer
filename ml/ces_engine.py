"""
HealDeal — Cost-Effectiveness Score (CES) Engine
Core ranking algorithm. Supports weighted multi-criteria scoring
with normalization, explainability, and sensitivity analysis.
"""
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin


# ---------- Haversine Distance ----------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two (lat, lon) points."""
    try:
        values = [float(lat1), float(lon1), float(lat2), float(lon2)]
        if not all(np.isfinite(value) for value in values):
            return float("nan")
        if not (-90 <= values[0] <= 90 and -90 <= values[2] <= 90
                and -180 <= values[1] <= 180 and -180 <= values[3] <= 180):
            return float("nan")
        lat1, lon1, lat2, lon2 = values
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371 * 2 * asin(min(1.0, a ** 0.5))
    except Exception:
        return float("nan")


# ---------- Normalization ----------
def min_max_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        median = s.median()
        s = s.fillna(float(median))
    else:
        s = pd.Series(0.5, index=s.index, dtype=float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


# ---------- CES Core ----------
class CESEngine:
    """
    Weighted CES = w_cost × cost_norm
                 + w_dist × dist_norm
                 + w_rating × (1 − rating_norm)   ← lower is better
                 + w_sent  × (1 − sent_norm)       ← lower is better

    Lower CES ≡ Better hospital.
    """

    DEFAULT_WEIGHTS = {
        "cost":      0.40,
        "rating":    0.30,
        "distance":  0.20,
        "sentiment": 0.10,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._normalize_weights()

    def _normalize_weights(self):
        clean = {}
        for key, default in self.DEFAULT_WEIGHTS.items():
            try:
                value = float(self.weights.get(key, default))
            except (TypeError, ValueError, OverflowError):
                value = default
            clean[key] = value if np.isfinite(value) and value >= 0 else default
        total = sum(clean.values())
        if total <= 0:
            clean = self.DEFAULT_WEIGHTS.copy()
            total = sum(clean.values())
        self.weights = {key: value / total for key, value in clean.items()}

    def score(self, df: pd.DataFrame, user_lat: float, user_lon: float,
              predicted_costs: np.ndarray = None) -> pd.DataFrame:
        """
        Score and rank hospitals. Returns enriched DataFrame sorted by CES ascending.
        """
        dff = df.copy().reset_index(drop=True)

        # Predicted cost column
        if predicted_costs is not None and len(predicted_costs) == len(dff):
            dff["pred_cost"] = np.maximum(pd.to_numeric(predicted_costs, errors="coerce"), 0)
        else:
            source_costs = pd.to_numeric(dff["avg_cost_for_procedure"], errors="coerce")
            median_cost = source_costs.median()
            dff["pred_cost"] = source_costs.fillna(float(median_cost) if pd.notna(median_cost) else 0).clip(lower=0)

        # Distance
        dff["distance_km"] = dff.apply(
            lambda r: haversine_km(user_lat, user_lon, r["latitude"], r["longitude"]),
            axis=1,
        )

        # Normalise sub-scores
        dff["cost_n"]   = min_max_norm(dff["pred_cost"])
        dff["rating_n"] = min_max_norm(dff["avg_rating"])
        dff["dist_n"]   = min_max_norm(dff["distance_km"])
        dff["sent_n"]   = min_max_norm(dff.get("sentiment_score", pd.Series(0.5, index=dff.index)))

        w = self.weights
        dff["CES"] = (
            w["cost"]      * dff["cost_n"]
            + w["distance"] * dff["dist_n"]
            + w["rating"]   * (1 - dff["rating_n"])
            + w["sentiment"]* (1 - dff["sent_n"])
        )

        # Percentile rank (0=best, 100=worst)
        dff["percentile_rank"] = dff["CES"].rank(pct=True).mul(100).round(1)

        return dff.sort_values("CES").reset_index(drop=True)

    def explain(self, row: pd.Series) -> dict:
        """Return per-criterion contribution breakdown for a single hospital row."""
        w = self.weights
        components = {
            "Cost":        w["cost"]       * float(row["cost_n"]),
            "Distance":    w["distance"]   * float(row["dist_n"]),
            "Rating":      w["rating"]     * float(1 - row["rating_n"]),
            "Sentiment":   w["sentiment"]  * float(1 - row["sent_n"]),
        }
        total = sum(components.values()) or 1
        pct = {k: round(v / total * 100, 1) for k, v in components.items()}
        dominant = max(components, key=components.get)
        tips = _generate_tips(dominant, row, pct)
        return {
            "hospital":   row.get("name", "Hospital"),
            "CES":        round(float(row["CES"]), 4),
            "components": components,
            "percentages": pct,
            "dominant_factor": dominant,
            "tips": tips,
        }

    def sensitivity_analysis(self, df: pd.DataFrame, user_lat: float,
                             user_lon: float, steps: int = 5) -> pd.DataFrame:
        """
        Vary cost weight 0→1 and observe how top-3 rankings change.
        Returns a DataFrame of results for each weight step.
        """
        records = []
        for w_cost in np.linspace(0, 1, steps):
            w_rem = 1 - w_cost
            weights = {
                "cost": w_cost,
                "rating": w_rem * 0.5,
                "distance": w_rem * 0.3,
                "sentiment": w_rem * 0.2,
            }
            eng = CESEngine(weights)
            scored = eng.score(df, user_lat, user_lon)
            top3 = scored.head(3)["name"].tolist()
            records.append({
                "cost_weight": round(w_cost, 2),
                "top1": top3[0] if len(top3) > 0 else "",
                "top2": top3[1] if len(top3) > 1 else "",
                "top3": top3[2] if len(top3) > 2 else "",
            })
        return pd.DataFrame(records)


def _generate_tips(dominant: str, row: pd.Series, pct: dict) -> list:
    tips = []
    if dominant == "Cost":
        tips.append(f"Cost drives {pct['Cost']}% of this score. Increasing your budget may open better-rated options.")
    if dominant == "Distance":
        tips.append(f"Distance is a major factor ({pct['Distance']}%). Expanding travel range could reveal cheaper hospitals.")
    if dominant == "Rating":
        tips.append(f"The generated sample rating contributes {pct['Rating']}% of this score; it is not a verified quality measure.")
    if dominant == "Sentiment":
        tips.append(f"Generated example-text sentiment contributes {pct['Sentiment']}% of this score; it is not patient feedback.")
    if float(row.get("avg_rating", 0)) >= 4.5:
        tips.append("The generated sample rating is high; it is not a verified satisfaction score.")
    if float(row.get("accredited", 0)) == 1.0 or str(row.get("accredited","")).lower() == "true":
        tips.append("This record has a generated sample accreditation flag, not a verified accreditation.")
    return tips
