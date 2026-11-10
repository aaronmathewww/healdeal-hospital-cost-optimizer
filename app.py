"""
HealDeal v2.0 — AI-Powered Healthcare Cost Optimizer
Educational demo application | not production-ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys, os, json, time

# ensure project root is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ml.ces_engine   import CESEngine, haversine_km
from ml.sentiment    import get_sentiment_label, extract_keywords, analyze_text
from utils.geocoder  import geocode_smart
from utils.auth      import register, login, get_user, add_xp, save_favorite, remove_favorite, append_search

# ─────────────────────────────────────────────
#  Config & Theme
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HealDeal — Healthcare Comparison Demo",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif; }

/* ── Root Palette ─────────────────────────────── */
:root {
    --primary:    #0D9488;   /* teal-600   */
    --primary-lt: #CCFBF1;   /* teal-100   */
    --accent:     #F59E0B;   /* amber-500  */
    --dark:       #0F172A;   /* slate-900  */
    --muted:      #64748B;   /* slate-500  */
    --card-bg:    #F8FAFC;
    --border:     #E2E8F0;
}

/* ── Header Banner ────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #134E4A 60%, #0D9488 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "💊";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.hero h1 { color: white; margin: 0; font-size: 2.2rem; }
.hero p  { color: #94A3B8; margin: 0.3rem 0 0; font-size: 1rem; }

/* ── Metric Cards ─────────────────────────────── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val  { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
.metric-card .lbl  { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

/* ── Hospital Cards ───────────────────────────── */
.hospital-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    transition: box-shadow 0.2s;
}
.hospital-card:hover { box-shadow: 0 4px 20px rgba(13,148,136,0.15); }
.hospital-card .rank {
    position: absolute; top: 1rem; right: 1rem;
    background: var(--primary); color: white;
    border-radius: 50%; width: 2rem; height: 2rem;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
}
.hospital-card h4 { margin: 0 0 0.3rem; color: var(--dark); font-size: 1.05rem; }
.hospital-card .tag {
    display: inline-block;
    background: var(--primary-lt); color: var(--primary);
    border-radius: 20px; padding: 2px 10px; font-size: 0.74rem;
    margin-right: 4px; margin-bottom: 4px;
}
.hospital-card .cost  { color: var(--primary); font-weight: 600; font-size: 1.1rem; }
.ces-bar-wrap { background:#E2E8F0; border-radius:8px; height:6px; margin-top:6px; }
.ces-bar      { background: var(--primary); border-radius:8px; height:6px; }

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A, #134E4A) !important;
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #CBD5E1 !important; }

/* ── Tab Styling ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-family: 'Sora', sans-serif;
}

/* ── Expander ─────────────────────────────────── */
.streamlit-expanderHeader { font-weight: 600; }

/* ── XP Badge ─────────────────────────────────── */
.xp-badge {
    display: inline-block;
    background: linear-gradient(135deg, #F59E0B, #D97706);
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
}

/* ── CES Gauge colours ───────────────────────── */
.ces-good   { color: #10B981; }
.ces-ok     { color: #F59E0B; }
.ces-poor   { color: #EF4444; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session State Defaults
# ─────────────────────────────────────────────
for key, val in {
    "logged_in": False, "username": "", "results_df": None,
    "last_search": {}, "compare_list": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = ROOT / "data" / "hospitals.csv"
    if not path.exists():
        st.error("Run `python data/generate_dataset.py` first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["avg_rating", "avg_cost_for_procedure", "latitude", "longitude",
              "bed_count", "num_reviews", "sentiment_score"]:
        if c not in df.columns:
            df[c] = 0
    df["accredited"] = df.get("accredited", False)
    return df

@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Train model if not yet trained, then load it."""
    from ml.cost_model import load_model, train
    try:
        return load_model()
    except Exception:
        train()
        return load_model()


# ─────────────────────────────────────────────
#  Hero Banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>HealDeal 💊</h1>
  <p>Healthcare cost comparison and decision-support demonstration</p>
</div>
""", unsafe_allow_html=True)
st.warning("DEMO DATA ONLY: hospital names, prices, ratings, reviews, accreditation, insurance tags and coordinates are synthetic. They do not represent real providers or quotations. Do not use for medical or financial decisions.")


# ─────────────────────────────────────────────
#  Sidebar — Auth + Filters
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ HealDeal")

    # Auth block
    if not st.session_state.logged_in:
        auth_mode = st.radio("", ["Login", "Register"], horizontal=True)
        uname = st.text_input("Username")
        pwd   = st.text_input("Password", type="password")
        if auth_mode == "Login":
            if st.button("Login", use_container_width=True):
                ok, msg, udata = login(uname, pwd)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            email = st.text_input("Email (optional)")
            if st.button("Create Account", use_container_width=True):
                ok, msg = register(uname, pwd, email)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        st.markdown("---")
        st.caption("💡 Login to save favorites, track XP & history")
    else:
        udata = get_user(st.session_state.username)
        xp    = udata.get("xp", 0)
        lvl   = udata.get("level", 1)
        st.markdown(f"👤 **{st.session_state.username}**")
        st.markdown(f'<span class="xp-badge">⚡ {xp} XP · Lv {lvl}</span>', unsafe_allow_html=True)
        xp_bar = min(xp % 100, 100)
        st.progress(xp_bar / 100, text=f"{xp_bar}/100 to next level")
        if udata.get("achievements"):
            with st.expander("🏅 Achievements"):
                for a in udata["achievements"]:
                    st.write(a)
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()
        st.markdown("---")

    st.markdown("### 🔍 Search Filters")

df = load_data()
specialties = ["Any"] + sorted(df["primary_specialty"].dropna().unique().tolist()) if not df.empty else ["Any"]
cities      = ["Any"] + sorted(df["city"].dropna().unique().tolist()) if not df.empty else ["Any"]
insurances  = ["Any", "Sample Insurance A", "Sample Insurance B", "Sample Insurance C",
               "Sample Insurance D", "Sample Insurance E"]
tiers       = ["Any", "Premium", "Mid-Range", "Budget"]

with st.sidebar:
    location_input  = st.text_input("📍 Your Location", placeholder="e.g. Delhi, Pune, Bengaluru")
    specialty_sel   = st.selectbox("🏥 Specialty", specialties)
    tier_sel        = st.selectbox("🏷️ Hospital Tier", tiers)
    insurance_sel   = st.selectbox("🛡️ Insurance", insurances)
    max_km          = st.slider("📏 Max Distance (km)", 5, 300, 50, 5)
    accredited_only = st.checkbox("Sample accreditation flag only")

    st.markdown("### ⚖️ Priority Weights")
    w_cost     = st.slider("💰 Cost",      0.0, 1.0, 0.40, 0.05)
    w_rating   = st.slider("⭐ Quality",   0.0, 1.0, 0.30, 0.05)
    w_distance = st.slider("📏 Distance",  0.0, 1.0, 0.20, 0.05)
    w_sentiment= st.slider("💬 Sentiment", 0.0, 1.0, 0.10, 0.05)

    use_ml_cost = st.checkbox("🤖 Use experimental ML cost estimate (synthetic data only)", value=False)
    search_btn  = st.button("🔍 Search Demo Records", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
#  Search Logic
# ─────────────────────────────────────────────
def run_search():
    if not location_input:
        st.warning("Please enter a location in the sidebar.")
        return

    with st.spinner("🔍 Resolving location and ranking hospitals..."):
        geo = geocode_smart(location_input)
        if geo is None:
            st.error("Could not resolve location. Try a major city name like 'Delhi' or 'Mumbai'.")
            return

        user_lat, user_lon = geo["lat"], geo["lon"]
        st.session_state.last_search = {
            "location": location_input,
            "display":  geo["display_name"],
            "lat": user_lat, "lon": user_lon,
        }

        dff = df.copy()

        # Filters
        if specialty_sel != "Any":
            dff = dff[dff["primary_specialty"].str.lower() == specialty_sel.lower()]
        if tier_sel != "Any":
            dff = dff[dff["tier"] == tier_sel]
        if accredited_only:
            dff = dff[dff["accredited"].astype(str).str.lower() == "true"]
        if insurance_sel != "Any":
            dff = dff[dff["insurance_accepted"].str.contains(insurance_sel, na=False)]

        # Distance filter
        dff["distance_km"] = dff.apply(
            lambda r: haversine_km(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1
        )
        dff = dff[dff["distance_km"] <= max_km].copy()

        if dff.empty:
            st.warning("No demo records found with these filters. Try relaxing distance or specialty.")
            return

        # ML cost prediction
        predicted_costs = None
        if use_ml_cost:
            try:
                from ml.cost_model import predict_cost
                predicted_costs = predict_cost(dff)
            except Exception as e:
                st.caption(f"ML model not ready ({e}), using CSV costs.")

        # CES scoring
        engine = CESEngine({"cost": w_cost, "rating": w_rating,
                            "distance": w_distance, "sentiment": w_sentiment})
        scored = engine.score(dff, user_lat, user_lon, predicted_costs)

        st.session_state.results_df = scored
        st.session_state.engine     = engine

        # XP & history
        if st.session_state.logged_in:
            add_xp(st.session_state.username, 10)
            append_search(st.session_state.username, {
                "location": location_input,
                "specialty": specialty_sel,
                "results": len(scored),
            })

if search_btn:
    run_search()


# ─────────────────────────────────────────────
#  Main Tabs
# ─────────────────────────────────────────────
tab_search, tab_analytics, tab_compare, tab_ml, tab_favs, tab_history, tab_about = st.tabs([
    "🔍 Results", "📊 Analytics", "⚖️ Compare", "🤖 ML Insights",
    "❤️ Favorites", "🕒 History", "ℹ️ About"
])


# ══════════════════════════════════════════════
#  TAB 1 · Results
# ══════════════════════════════════════════════
with tab_search:
    if st.session_state.results_df is None:
        st.info("👈 Enter a city and click **Search Demo Records** to get started.")
        # Quick stats
        if not df.empty:
            st.markdown("### 🗺️ Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="val">{len(df):,}</div><div class="lbl">Synthetic demo records</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="val">{df["city"].nunique()}</div><div class="lbl">Cities Covered</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="val">{df["primary_specialty"].nunique()}</div><div class="lbl">Specialties</div></div>', unsafe_allow_html=True)
            with c4:
                avg_r = df["avg_rating"].mean()
                st.markdown(f'<div class="metric-card"><div class="val">{avg_r:.1f}⭐</div><div class="lbl">Average sample rating</div></div>', unsafe_allow_html=True)
    else:
        scored = st.session_state.results_df
        engine = st.session_state.engine
        geo    = st.session_state.last_search

        st.success(f"📍 **{geo['display_name']}** · {len(scored)} synthetic demo records found")

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (len(scored), "Synthetic records found"),
            (f"₹{scored['pred_cost'].min():,.0f}", "Lowest sample cost"),
            (f"₹{scored['pred_cost'].mean():,.0f}", "Average sample cost"),
            (f"{scored['avg_rating'].max():.1f}⭐", "Highest sample rating"),
            (f"{scored['distance_km'].min():.1f} km", "Nearest"),
        ]
        for col, (val, lbl) in zip([c1, c2, c3, c4, c5], metrics):
            with col:
                st.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        max_top_n = min(50, len(scored))
        top_n = st.slider("Show top N results", 1, max_top_n, min(10, max_top_n), 1)
        top   = scored.head(top_n)

        # Hospital cards + map side by side
        col_cards, col_map = st.columns([1, 1])

        with col_cards:
            st.markdown("#### 🏆 Ranked Synthetic Demo Records")
            for i, row in top.iterrows():
                ces_pct = (1 - row["CES"]) * 100
                acc_badge = "Sample accreditation flag" if str(row.get("accredited", "")).lower() == "true" else ""
                sentiment_lbl = get_sentiment_label(float(row["sentiment_score"]))
                ins = str(row.get("insurance_accepted","")).split("|")[:2]
                ins_tags = "".join([f'<span class="tag">Sample insurance tag: {x.strip()}</span>' for x in ins if x.strip()])
                st.markdown(f"""
<div class="hospital-card">
  <div class="rank">#{i+1}</div>
  <h4>{row['name']} {acc_badge}</h4>
  <span class="tag">📍 {row['city']}</span>
  <span class="tag">🏥 {row['primary_specialty']}</span>
  <span class="tag">🏷️ {row.get('tier','')}</span>
  {ins_tags}
  <br><br>
  <span class="cost">₹{row['pred_cost']:,.0f}</span> <small>Sample / experimental cost</small>
  &nbsp;&nbsp; ⭐ Sample rating {row['avg_rating']:.1f}
  &nbsp;&nbsp; 📏 {row['distance_km']:.1f} km
  &nbsp;&nbsp; {sentiment_lbl}
  <div class="ces-bar-wrap"><div class="ces-bar" style="width:{ces_pct:.0f}%"></div></div>
  <small style="color:#64748B">Demo CES: {row['CES']:.4f} · Percentile within this result set: {row['percentile_rank']:.0f}th</small>
</div>""", unsafe_allow_html=True)

                with st.expander(f"💡 Explain ranking for {row['name']}"):
                    expl = engine.explain(row)
                    exp_df = pd.DataFrame({
                        "Factor":    list(expl["percentages"].keys()),
                        "% Weight":  list(expl["percentages"].values()),
                    })
                    fig_pie = px.pie(exp_df, names="Factor", values="% Weight",
                                     hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=220)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    for tip in expl["tips"]:
                        st.markdown(f"- {tip}")
                    if "reviews" in row and str(row["reviews"]).strip():
                        kws = extract_keywords(str(row["reviews"]))
                        st.caption(f"🔑 Key words: {', '.join(kws)}")

                if st.session_state.logged_in:
                    if st.button(f"❤️ Save #{i+1}", key=f"fav_{i}"):
                        ok, msg = save_favorite(st.session_state.username, {
                            "hospital_id": row["hospital_id"],
                            "name": row["name"],
                            "city": row["city"],
                            "pred_cost": float(row["pred_cost"]),
                            "avg_rating": float(row["avg_rating"]),
                            "specialty": row["primary_specialty"],
                        })
                        if ok:
                            add_xp(st.session_state.username, 30)
                            st.success("Saved! +30 XP")
                        else:
                            st.info(msg)

        with col_map:
            st.markdown("#### 🗺️ Simulated Map View")
            st.caption("Map points are generated around city centres and do not represent real hospital locations.")
            map_df = top.rename(columns={"latitude": "lat", "longitude": "lon"}).copy()
            map_df = map_df[["lat", "lon", "name", "pred_cost", "avg_rating", "distance_km"]].dropna()
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", hover_name="name",
                hover_data={"pred_cost": True, "avg_rating": True, "distance_km": True,
                            "lat": False, "lon": False},
                color="avg_rating", size="pred_cost",
                color_continuous_scale="teal",
                size_max=18, zoom=10,
                mapbox_style="open-street-map",
            )
            fig_map.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=600)
            st.plotly_chart(fig_map, use_container_width=True)

        # Raw data table
        with st.expander("📋 Full Data Table"):
            show_cols = ["hospital_id", "name", "city", "primary_specialty", "tier",
                         "pred_cost", "avg_rating", "distance_km", "sentiment_score",
                         "bed_count", "accredited", "CES", "percentile_rank"]
            show_cols = [c for c in show_cols if c in scored.columns]
            st.dataframe(scored[show_cols], use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 · Analytics
# ══════════════════════════════════════════════
with tab_analytics:
    if st.session_state.results_df is None:
        st.info("Run a search first to see analytics.")
    else:
        scored = st.session_state.results_df

        st.markdown("### 📊 Search Results Analytics")

        # Row 1 — cost dist + rating vs cost scatter
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig_hist = px.histogram(scored, x="pred_cost", nbins=20,
                                    title="Cost Distribution",
                                    color_discrete_sequence=["#0D9488"])
            fig_hist.update_layout(bargap=0.1, xaxis_title="Sample / experimental cost (₹)",
                                   yaxis_title="Hospitals", height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

        with r1c2:
            fig_sc = px.scatter(scored, x="pred_cost", y="avg_rating",
                                color="tier", size="bed_count",
                                hover_name="name",
                                title="Cost vs Quality",
                                color_discrete_map={
                                    "Premium": "#F59E0B",
                                    "Mid-Range": "#0D9488",
                                    "Budget": "#6366F1"
                                })
            fig_sc.update_layout(height=320)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Row 2 — sentiment dist + CES bar
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig_sent = px.histogram(scored, x="sentiment_score", nbins=15,
                                    title="Sentiment Score Distribution",
                                    color_discrete_sequence=["#6366F1"])
            fig_sent.update_layout(height=320)
            st.plotly_chart(fig_sent, use_container_width=True)

        with r2c2:
            top10 = scored.head(10)
            fig_ces = px.bar(top10, x="name", y="CES",
                             title="Top 10 by CES (lower = better)",
                             color="CES",
                             color_continuous_scale="teal_r")
            fig_ces.update_layout(xaxis_tickangle=-40, height=320)
            st.plotly_chart(fig_ces, use_container_width=True)

        # Row 3 — tier breakdown pie + distance histogram
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            tier_counts = scored["tier"].value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            fig_pie = px.pie(tier_counts, names="tier", values="count",
                             title="Hospital Tier Breakdown",
                             hole=0.4,
                             color_discrete_map={
                                 "Premium": "#F59E0B",
                                 "Mid-Range": "#0D9488",
                                 "Budget": "#6366F1"
                             })
            fig_pie.update_layout(height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        with r3c2:
            fig_dist = px.histogram(scored, x="distance_km", nbins=20,
                                    title="Distance Distribution (km)",
                                    color_discrete_sequence=["#F59E0B"])
            fig_dist.update_layout(height=320)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Sensitivity analysis
        if len(scored) >= 5:
            st.markdown("### 🎯 CES Sensitivity Analysis")
            st.caption("How does changing cost priority affect top-3 rankings?")
            engine = st.session_state.engine
            geo    = st.session_state.last_search
            sa = engine.sensitivity_analysis(df.copy(), geo["lat"], geo["lon"], steps=6)
            st.dataframe(sa, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 · Compare
# ══════════════════════════════════════════════
with tab_compare:
    if st.session_state.results_df is None:
        st.info("Run a search first, then select hospitals to compare.")
    else:
        scored = st.session_state.results_df
        all_names = scored["name"].tolist()
        selected  = st.multiselect("Select hospitals to compare (2–4)", all_names,
                                   max_selections=4,
                                   default=all_names[:min(3, len(all_names))])
        if len(selected) >= 2:
            cmp_df = scored[scored["name"].isin(selected)].copy()

            # Radar chart
            categories = ["Cost (inv)", "Rating", "Distance (inv)", "Sentiment", "Bed Count"]
            fig_radar = go.Figure()
            for _, row in cmp_df.iterrows():
                vals = [
                    1 - float(row["cost_n"]),
                    float(row["rating_n"]),
                    1 - float(row["dist_n"]),
                    float(row["sent_n"]),
                    float(row["bed_count"]) / cmp_df["bed_count"].max() if cmp_df["bed_count"].max() > 0 else 0.5,
                ]
                vals += [vals[0]]     # close the loop
                cats  = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=row["name"][:30],
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Hospital Comparison Radar",
                height=480,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Tabular comparison
            st.markdown("#### Side-by-Side Comparison")
            compare_cols = ["name", "city", "tier", "pred_cost", "avg_rating",
                            "distance_km", "sentiment_score", "bed_count",
                            "accredited", "primary_procedure", "CES"]
            compare_cols = [c for c in compare_cols if c in cmp_df.columns]
            st.dataframe(cmp_df[compare_cols].set_index("name").T, use_container_width=True)
        else:
            st.info("Select at least 2 hospitals.")


# ══════════════════════════════════════════════
#  TAB 4 · ML Insights
# ══════════════════════════════════════════════
with tab_ml:
    st.markdown("### 🤖 Machine Learning Insights")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### Cost Prediction Model")
        with st.spinner("Loading model metrics..."):
            try:
                from ml.cost_model import get_feature_importance, load_model, train, prepare_features
                model, encoders = load_model()
                fi = get_feature_importance()
                if not fi.empty:
                    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                                    title="Feature Importances (Random Forest)",
                                    color="importance",
                                    color_continuous_scale="teal")
                    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=360)
                    st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.warning(f"Experimental model not yet trained. It is evaluated only on synthetic demonstration data.")
                if st.button("🚀 Train Cost Prediction Model"):
                    with st.spinner("Training experimental Random Forest model..."):
                        from ml.cost_model import train
                        model, encoders, metrics = train()
                        st.success(f"Trained on synthetic data only · Holdout R²={metrics['holdout_r2']:.4f}, MAE=₹{metrics['mae_inr']:,.0f}")
                        st.rerun()

    with col_m2:
        st.markdown("#### Sentiment Analysis Demo")
        test_review = st.text_area("Enter example text to analyse (not a real patient review):",
                                   value="The doctors were very professional and the facilities were clean. Highly recommend!")
        if st.button("🔍 Analyse Sentiment"):
            result = analyze_text(test_review)
            score  = result["score"]
            label  = get_sentiment_label(score)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Sentiment Score", f"{score:.3f}")
                st.markdown(f"**Label:** {label}")
                st.markdown(f"**Compound:** {result['compound']:.3f}")
            with col_b:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    title={"text": "Sentiment %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#0D9488"},
                        "steps": [
                            {"range": [0, 40],  "color": "#FEE2E2"},
                            {"range": [40, 60], "color": "#FEF9C3"},
                            {"range": [60, 100],"color": "#D1FAE5"},
                        ],
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

    # Dataset-wide ML stats
    st.markdown("#### 📈 Dataset-Wide Sentiment Distribution")
    if not df.empty:
        sent_bins = pd.cut(df["sentiment_score"],
                           bins=[0, 0.4, 0.55, 0.7, 1.0],
                           labels=["Negative", "Neutral", "Positive", "Very Positive"])
        sent_dist = sent_bins.value_counts().reset_index()
        sent_dist.columns = ["label", "count"]
        fig_sent = px.bar(sent_dist, x="label", y="count",
                          color="label",
                          color_discrete_map={
                              "Very Positive": "#10B981",
                              "Positive":      "#0D9488",
                              "Neutral":       "#F59E0B",
                              "Negative":      "#EF4444",
                          },
                          title="Generated Review-Text Sentiment Across Demo Records")
        fig_sent.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_sent, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 · Favorites
# ══════════════════════════════════════════════
with tab_favs:
    if not st.session_state.logged_in:
        st.info("Please login to view favorites.")
    else:
        udata = get_user(st.session_state.username)
        favs  = udata.get("favorites", [])
        st.markdown(f"### ❤️ Your Favorites ({len(favs)})")
        if favs:
            fav_df = pd.DataFrame(favs)
            st.dataframe(fav_df, use_container_width=True)
            rm_id = st.selectbox("Remove hospital", ["—"] + [f["hospital_id"] for f in favs])
            if st.button("🗑️ Remove") and rm_id != "—":
                remove_favorite(st.session_state.username, rm_id)
                st.success("Removed.")
                st.rerun()
        else:
            st.info("No favorites yet. Save hospitals from the Results tab.")


# ══════════════════════════════════════════════
#  TAB 6 · History
# ══════════════════════════════════════════════
with tab_history:
    if not st.session_state.logged_in:
        st.info("Please login to view search history.")
    else:
        udata = get_user(st.session_state.username)
        hist  = udata.get("search_history", [])
        st.markdown(f"### 🕒 Search History (last {min(len(hist), 20)})")
        if hist:
            st.dataframe(pd.DataFrame(hist[:20]), use_container_width=True)
        else:
            st.info("No searches yet.")


# ══════════════════════════════════════════════
#  TAB 7 · About
# ══════════════════════════════════════════════
with tab_about:
    st.markdown("""
### 💊 HealDeal v2.0 — AI-Powered Healthcare Cost Optimizer

**B.Tech Capstone Project** · Computer Science & Engineering

---

#### 🧠 How CES Works

The **Cost-Effectiveness Score** ranks hospitals using a weighted multi-criteria formula:

```
CES = w₁×cost_norm + w₂×dist_norm + w₃×(1−rating_norm) + w₄×(1−sentiment_norm)
```

Lower CES = Better hospital. All sub-scores are min-max normalised to [0,1].

---

#### 🏗️ Architecture

| Layer | Technology |
|---|---|
| Primary Frontend UI | HTML/CSS/JavaScript served by Flask |
| Alternate Frontend UI | Streamlit 1.x |
| ML Cost Model | Random Forest Regressor (experimental, synthetic data only) |
| Sentiment Analysis | VADER NLP |
| Geocoding | Nominatim (OpenStreetMap) |
| Data Storage | CSV + JSON (file-based) |
| Distance Calc | Haversine Formula |

---

#### 📁 Project Structure
```
healdeal/
├── backend/app.py           ← Primary Flask API + frontend host
├── frontend/index.html      ← Browser SPA
├── app.py                   ← Optional Streamlit interface
├── wsgi.py                  ← WSGI entry point
├── ml/
│   ├── ces_engine.py        ← CES scoring algorithm
│   ├── cost_model.py        ← Experimental ML cost model
│   └── sentiment.py         ← NLP sentiment analysis (VADER)
├── utils/
│   ├── auth.py              ← User authentication & XP system
│   └── geocoder.py          ← Location resolution with caching
├── data/
│   ├── hospitals.csv        ← Synthetic demonstration dataset
│   └── generate_dataset.py  ← Synthetic data generator
├── models/                  ← Trained ML model artifacts
└── requirements.txt
```
""")

    if not df.empty:
        st.markdown("#### 📊 Dataset Sample")
        st.dataframe(df.sample(5), use_container_width=True)
