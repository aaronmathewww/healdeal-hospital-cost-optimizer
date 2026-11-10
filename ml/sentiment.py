"""
HealDeal — Sentiment Analysis Module
Analyzes example review text with VADER when installed and a small rule-based
lexicon fallback otherwise. Bundled review snippets are synthetic examples.
"""
import re
import numpy as np
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    BACKEND = "vader"
except ImportError:
    _analyzer = None
    BACKEND = "rule"


# ---------- Healthcare-domain lexicon boost ----------
POSITIVE_TERMS = {
    "excellent", "amazing", "outstanding", "professional", "skilled", "caring",
    "clean", "hygienic", "modern", "efficient", "timely", "affordable",
    "world-class", "compassionate", "knowledgeable", "expert", "quick",
    "friendly", "attentive", "transparent", "satisfied", "recommend",
    "recovered", "healed", "comfortable", "safe", "trust",
}
NEGATIVE_TERMS = {
    "dirty", "rude", "expensive", "overpriced", "slow", "negligent",
    "unhygienic", "careless", "incompetent", "waiting", "delay",
    "unhelpful", "cold", "dismissive", "overcharged", "billing",
    "crowded", "unresponsive", "dangerous", "wrong", "mistake",
}


def _rule_based_score(text: str) -> float:
    """Simple rule-based fallback sentiment scorer."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_TERMS)
    neg = len(words & NEGATIVE_TERMS)
    total = pos + neg
    if total == 0:
        return 0.5
    return round((pos / total) * 0.7 + 0.15, 3)   # map to [0.15, 0.85]


def analyze_text(text: str) -> dict:
    """
    Analyse a single review text.
    Returns dict with keys: compound, label, score (0-1 normalised).
    """
    if not text or str(text).strip() == "":
        return {"compound": 0.0, "label": "neutral", "score": 0.5}

    if BACKEND == "vader":
        scores = _analyzer.polarity_scores(str(text))
        compound = scores["compound"]                    # −1 to +1
        score = round((compound + 1) / 2, 3)            # map to 0-1
        label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        return {"compound": compound, "label": label, "score": score}
    else:
        score = _rule_based_score(str(text))
        compound = round(score * 2 - 1, 3)
        label = "positive" if score > 0.55 else ("negative" if score < 0.45 else "neutral")
        return {"compound": compound, "label": label, "score": score}


def analyze_reviews_column(reviews_series: pd.Series) -> pd.Series:
    """
    Takes a Series of pipe-separated reviews per hospital,
    returns a Series of aggregate sentiment scores (0-1).
    """
    def agg_score(reviews_str):
        if pd.isna(reviews_str) or str(reviews_str).strip() == "":
            return 0.5
        parts = str(reviews_str).split("|")
        scores = [analyze_text(r)["score"] for r in parts if r.strip()]
        return round(np.mean(scores), 3) if scores else 0.5

    return reviews_series.apply(agg_score)


def get_sentiment_label(score: float) -> str:
    if score >= 0.7:   return "🟢 Very Positive"
    elif score >= 0.55: return "🟡 Positive"
    elif score >= 0.45: return "⚪ Neutral"
    elif score >= 0.3:  return "🟠 Negative"
    else:               return "🔴 Very Negative"


def extract_keywords(reviews_str: str, top_n: int = 5) -> list:
    """Extract most frequent meaningful words from reviews."""
    stopwords = {"the","a","an","is","was","to","and","of","in","it","i",
                 "for","that","this","was","are","were","have","had","my","me",
                 "they","their","with","very","but","not","so","be","on","at",
                 "by","from","or","we","our","us","as","he","she","his","her"}
    words = re.findall(r"\b[a-z]{4,}\b", reviews_str.lower())
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]


if __name__ == "__main__":
    test_reviews = [
        "Excellent doctors and very clean facilities. Highly recommend!",
        "Overcrowded and the billing was confusing. Long waiting time.",
        "Average experience. Nothing exceptional but treatment was okay.",
    ]
    for r in test_reviews:
        result = analyze_text(r)
        print(f"{get_sentiment_label(result['score'])} | score={result['score']} | {r[:60]}")
