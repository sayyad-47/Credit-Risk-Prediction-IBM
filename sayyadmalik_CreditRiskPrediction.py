"""
Credit Risk & Loan Default Prediction
AICTE / IBM SkillsBuild Academic Internship
Author : Sayyad Malik
Dataset: German Credit Dataset (Cleaned)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk & Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL THEME CSS — forces light mode, fixes all text/overlap issues
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════
   DEEP-SPACE THEME — dark navy canvas + electric teal
   ══════════════════════════════════════════════════════ */

/* ── Global background & text ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container,
section.main > div {
    background-color: #0a0f1e !important;
    color: #e0e8ff !important;
}

/* ── Root font ── */
html, body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
* {
    box-sizing: border-box;
}

/* ── Block container padding ── */
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar — deep void with teal accent line ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050a14 0%, #0d1b2a 60%, #091624 100%) !important;
    border-right: 2px solid #00d4aa !important;
}
[data-testid="stSidebar"] * { color: #b0c8e8 !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #88acd0 !important;
    font-size: 0.93rem !important;
    padding: 6px 4px !important;
    border-radius: 6px;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #00d4aa !important;
    background-color: rgba(0,212,170,0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #5a7a9a !important; }
[data-testid="stSidebar"] hr { border-color: #1a3050 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] { display: none; }

/* ── KPI Metric Cards — dark glass card ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%) !important;
    border: 1px solid #1e3a5f !important;
    border-top: 2px solid #00d4aa !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.08) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
[data-testid="metric-container"] label {
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: #5a8aaa !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #00d4aa !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    color: #5a8aaa !important;
}

/* ── Plotly chart containers — dark glass ── */
[data-testid="stPlotlyChart"] {
    background: #0d1b2a !important;
    border-radius: 14px !important;
    border: 1px solid #1e3a5f !important;
    padding: 0.5rem !important;
    box-shadow: 0 4px 24px rgba(0,212,170,0.07) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    border-left-width: 4px !important;
    background-color: #0d1b2a !important;
    color: #e0e8ff !important;
}

/* ── Form inputs ── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div > div {
    background-color: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #e0e8ff !important;
}
[data-testid="stWidgetLabel"] p { color: #88acd0 !important; }
[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input { color: #e0e8ff !important; }

/* ── Form submit button — teal glow ── */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #00b894, #00d4aa) !important;
    color: #050a14 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.05em;
    box-shadow: 0 4px 18px rgba(0,212,170,0.40) !important;
    transition: opacity 0.2s;
}
[data-testid="stFormSubmitButton"] button:hover { opacity: 0.85 !important; }

/* ── Markdown text ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #b0c8e8 !important; }

/* ── Slider track ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #5a8aaa !important; }

/* ── Dividers ── */
hr { border-color: #1a3050 !important; }

/* ── Section headers rendered via st.markdown plain text ── */
h3 { color: #00d4aa !important; margin-bottom: 0 !important; }

/* ── Hide Streamlit footer & menu ── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "german_credit_dataset_cleaned.csv"

@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Default"] = (df["Risk"] == "Risk").astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose",
    "LoanAmount_INR", "ExistingSavings", "EmploymentDuration",
    "InstallmentPercent", "Sex", "OthersOnLoan", "CurrentResidenceDuration",
    "OwnsProperty", "Age", "InstallmentPlans", "Housing",
    "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker",
]

@st.cache_resource(show_spinner="Training model …")
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df["Default"]

    label_encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.85, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
        "report":   classification_report(y_test, y_pred, output_dict=True),
        "cm":       confusion_matrix(y_test, y_pred),
        "feature_importance": pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False),
    }
    return model, label_encoders, metrics

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_risk(model, label_encoders, input_dict: dict) -> float:
    row = pd.DataFrame([input_dict])
    for col in row.select_dtypes(include="object").columns:
        if col in label_encoders:
            le = label_encoders[col]
            val = row[col].iloc[0]
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = le.transform([val])
    return float(model.predict_proba(row[FEATURE_COLS])[0, 1])

# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "blue":    "#00d4aa",   # electric teal  — "No Risk"
    "red":     "#ff4d6d",   # neon coral     — "Risk"
    "green":   "#00e676",
    "orange":  "#ff9f43",
    "purple":  "#a29bfe",
    "cyan":    "#18dcff",
    "yellow":  "#ffd32a",
}
CHART_COLORS = list(PALETTE.values())
COLOR_MAP = {"No Risk": PALETTE["blue"], "Risk": PALETTE["red"]}

# Dark-theme chart backgrounds
_BG   = "#0d1b2a"   # chart paper / canvas
_PLOT = "#0a1628"   # inner plot area
_GRID = "#1e3a5f"   # grid lines
_TEXT = "#c0d8f0"   # axis labels & titles

def _chart(fig, height: int = 420, tickangle: int = -30) -> go.Figure:
    """Apply deep-space theme to any Plotly 2D figure."""
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT,
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color=_TEXT),
        height=height,
        margin=dict(l=55, r=35, t=55, b=100),
        legend=dict(
            bgcolor="#0d1b2a",
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(size=12, color=_TEXT),
        ),
        xaxis=dict(
            tickangle=tickangle,
            gridcolor=_GRID,
            linecolor=_GRID,
            tickfont=dict(color=_TEXT, size=11),
            title_font=dict(color=_TEXT),
        ),
        yaxis=dict(
            gridcolor=_GRID,
            linecolor=_GRID,
            tickfont=dict(color=_TEXT, size=11),
            title_font=dict(color=_TEXT),
        ),
        title_font=dict(size=14, color="#00d4aa", family="Segoe UI, sans-serif"),
    )
    fig.update_traces(marker_line_color=_BG, marker_line_width=0.8)
    return fig


def _section(title: str):
   
    # Write the title as plain markdown so emojis are rendered by the browser,
    # then draw the accent underline as a thin HTML div below it.
    st.markdown(title, unsafe_allow_html=False)
    st.markdown(
        "<div style='border-bottom:2px solid #00d4aa;"
        "margin:-0.55rem 0 0.7rem;'></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()
model, label_encoders, metrics = train_model(df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:16px 0 20px;'>"
        "<div style='font-size:2.4rem;margin-bottom:6px;'>🏦</div>"
        "<div style='font-weight:800;font-size:1.05rem;color:#ffffff;'>"
        "Credit Risk AI</div>"
        "<div style='font-size:0.70rem;color:#7e9bc0;margin-top:4px;'>"
        "AICTE / IBM SkillsBuild Internship</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigation",
        options=[
            "📊  Executive Overview",
            "📂  Portfolio Analysis",
            "🔍  Customer Risk Predictor",
            "🤖  Model Performance",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<div style='font-size:0.76rem;color:#7e9bc0;line-height:1.9;padding:0 4px;'>"
        f"<b style='color:#a0b8d0;'>Dataset:</b> German Credit (Cleaned)<br>"
        f"<b style='color:#a0b8d0;'>Records:</b> {len(df):,}<br>"
        f"<b style='color:#a0b8d0;'>Model:</b> Gradient Boosting<br>"
        f"<b style='color:#a0b8d0;'>AUC-ROC:</b> {metrics['roc_auc']:.3f}<br>"
        f"<b style='color:#a0b8d0;'>Accuracy:</b> {metrics['accuracy']:.1%}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:0.68rem;color:#4a6080;padding:0 4px;'>"
        "Author: Sayyad Malik<br>IBM SkillsBuild · 2024</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:linear-gradient(90deg,#050a14 0%,#0d2b3e 50%,#003d2e 100%);"
    "color:#e0e8ff;padding:1.1rem 1.6rem;border-radius:14px;margin-bottom:1rem;"
    "border:1px solid #00d4aa;"
    "box-shadow:0 0 32px rgba(0,212,170,0.18),0 4px 16px rgba(0,0,0,0.5);'>"
    "<div style='font-size:1.45rem;font-weight:800;letter-spacing:0.01em;color:#ffffff;'>"
    "🏦 Credit Risk &amp; Loan Default Prediction</div>"
    "<div style='font-size:0.82rem;color:#00d4aa;margin-top:5px;letter-spacing:0.03em;'>"
    "AICTE / IBM SkillsBuild Academic Internship &nbsp;·&nbsp; "
    "German Credit Dataset &nbsp;·&nbsp; Gradient Boosting Classifier</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊  Executive Overview":

    _section("📌 Portfolio KPIs")

    total        = len(df)
    default_n    = int(df["Default"].sum())
    default_rate = df["Default"].mean()
    avg_dur      = df["LoanDuration"].mean()
    avg_amt      = df["LoanAmount_INR"].mean()
    avg_age      = df["Age"].mean()

    # 2 rows of 3 cards each — prevents label truncation at 6-column width
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    row1_c1.metric("Total Customers",   f"{total:,}")
    row1_c2.metric("Defaulters",        f"{default_n:,}")
    row1_c3.metric("Default Rate",      f"{default_rate:.1%}")
    row2_c1.metric("Avg Loan Duration", f"{avg_dur:.1f} mo")
    row2_c2.metric("Avg Loan Amount",   f"₹{avg_amt:,.0f}")
    row2_c3.metric("Avg Customer Age",  f"{avg_age:.1f} yrs")

    _section("📊 Default Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        counts = df["Risk"].value_counts().reset_index()
        counts.columns = ["Risk Category", "Count"]
        fig_pie = px.pie(
            counts, names="Risk Category", values="Count",
            color="Risk Category", color_discrete_map=COLOR_MAP,
            title="Overall Risk Distribution", hole=0.44,
        )
        fig_pie.update_layout(
            paper_bgcolor=_BG,
            font=dict(size=12, color=_TEXT),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(bgcolor=_BG, bordercolor=_GRID, borderwidth=1,
                        font=dict(color=_TEXT)),
            title_font=dict(size=14, color="#00d4aa"),
        )
        fig_pie.update_traces(textinfo="percent+label", textfont=dict(size=13, color="#ffffff"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        dur_df = df.copy()
        dur_df["Duration Band"] = pd.cut(
            dur_df["LoanDuration"],
            bins=[0, 6, 12, 24, 36, 48, 100],
            labels=["≤6 mo", "7-12 mo", "13-24 mo", "25-36 mo", "37-48 mo", "49+ mo"],
        )
        band_risk = (
            dur_df.groupby("Duration Band", observed=True)["Default"]
            .mean().reset_index()
            .rename(columns={"Default": "Default Rate"})
        )
        band_risk["Default Rate %"] = (band_risk["Default Rate"] * 100).round(2)
        fig_bar = px.bar(
            band_risk, x="Duration Band", y="Default Rate %",
            color="Default Rate %",
            color_continuous_scale=["#c8d9f5", "#0f52ba", "#e03131"],
            title="Default Rate by Loan Duration Band",
            text="Default Rate %",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                               textfont=dict(color="#1a1a2e"))
        _chart(fig_bar, height=420, tickangle=-35)
        # Extra bottom margin so rotated labels don't clip
        fig_bar.update_layout(margin=dict(l=55, r=35, t=55, b=130))
        st.plotly_chart(fig_bar, use_container_width=True)

    _section("💰 Loan Amount Distribution — 3D Surface")
    # Build 2D histogram data then render as a draggable 3D surface
    _amt_bins  = np.linspace(df["LoanAmount_INR"].min(), df["LoanAmount_INR"].max(), 35)
    _dur_bins  = np.linspace(df["LoanDuration"].min(),   df["LoanDuration"].max(),   20)
    _h2d, _xedge, _yedge = np.histogram2d(
        df["LoanAmount_INR"], df["LoanDuration"], bins=[_amt_bins, _dur_bins]
    )
    fig_surf = go.Figure(go.Surface(
        z=_h2d.T,
        x=(_xedge[:-1] + _xedge[1:]) / 2,
        y=(_yedge[:-1] + _yedge[1:]) / 2,
        colorscale=[[0, "#003d2e"], [0.5, "#00d4aa"], [1, "#ff4d6d"]],
        opacity=0.88,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#00d4aa", project_z=True)),
    ))
    fig_surf.update_layout(
        paper_bgcolor=_BG,
        scene=dict(
            xaxis=dict(title="Loan Amount (₹)", tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            yaxis=dict(title="Duration (mo)",   tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            zaxis=dict(title="Count",           tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1)),
        ),
        title=dict(text="Loan Amount × Duration — 3D Frequency Surface (drag to rotate)",
                   font=dict(size=14, color="#00d4aa")),
        height=480,
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(color=_TEXT),
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    _section("👤 Age vs Loan Amount — 3D Scatter")
    _samp = df.sample(min(800, len(df)), random_state=7)
    fig_3d = go.Figure()
    for risk_val, color in [("No Risk", PALETTE["blue"]), ("Risk", PALETTE["red"])]:
        _sub = _samp[_samp["Risk"] == risk_val]
        fig_3d.add_trace(go.Scatter3d(
            x=_sub["Age"],
            y=_sub["LoanAmount_INR"],
            z=_sub["LoanDuration"],
            mode="markers",
            name=risk_val,
            marker=dict(
                size=4,
                color=color,
                opacity=0.75,
                line=dict(width=0.3, color="#ffffff"),
            ),
            hovertemplate=(
                "<b>" + risk_val + "</b><br>"
                "Age: %{x}<br>"
                "Amount: ₹%{y:,.0f}<br>"
                "Duration: %{z} mo<extra></extra>"
            ),
        ))
    fig_3d.update_layout(
        paper_bgcolor=_BG,
        scene=dict(
            xaxis=dict(title="Age",             tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            yaxis=dict(title="Loan Amount (₹)", tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            zaxis=dict(title="Duration (mo)",   tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        title=dict(text="Age × Loan Amount × Duration — 3D Scatter (drag to rotate)",
                   font=dict(size=14, color="#00d4aa")),
        legend=dict(bgcolor=_BG, bordercolor=_GRID, borderwidth=1,
                    font=dict(color=_TEXT)),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(color=_TEXT),
    )
    st.plotly_chart(fig_3d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📂  Portfolio Analysis":

    _section("🏠 Risk by Housing & Loan Purpose")
    col1, col2 = st.columns(2)

    with col1:
        housing_risk = (
            df.groupby("Housing")["Default"]
            .mean().reset_index()
            .rename(columns={"Default": "Default Rate"})
        )
        housing_risk["Default Rate %"] = (housing_risk["Default Rate"] * 100).round(2)
        fig_h = px.bar(
            housing_risk.sort_values("Default Rate %", ascending=False),
            x="Housing", y="Default Rate %",
            color="Default Rate %",
            color_continuous_scale=["#c8d9f5", "#0f52ba", "#e03131"],
            title="Default Rate by Housing Type", text="Default Rate %",
        )
        fig_h.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             textfont=dict(color="#1a1a2e"))
        _chart(fig_h, tickangle=-20)
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        purpose_risk = (
            df.groupby("LoanPurpose")["Default"]
            .mean().reset_index()
            .rename(columns={"Default": "Default Rate"})
            .sort_values("Default Rate", ascending=False)
        )
        purpose_risk["Default Rate %"] = (purpose_risk["Default Rate"] * 100).round(2)
        fig_p = px.bar(
            purpose_risk, x="LoanPurpose", y="Default Rate %",
            color="Default Rate %",
            color_continuous_scale=["#c8d9f5", "#0f52ba", "#e03131"],
            title="Default Rate by Loan Purpose", text="Default Rate %",
        )
        fig_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             textfont=dict(color="#1a1a2e"))
        _chart(fig_p, tickangle=-45)
        st.plotly_chart(fig_p, use_container_width=True)

    _section("🏦 Risk by Checking Status & Savings")
    col3, col4 = st.columns(2)

    with col3:
        chk_risk = (
            df.groupby("CheckingStatus")["Default"]
            .mean().reset_index()
            .rename(columns={"Default": "Default Rate"})
        )
        chk_risk["Default Rate %"] = (chk_risk["Default Rate"] * 100).round(2)
        fig_c = px.bar(
            chk_risk.sort_values("Default Rate %", ascending=False),
            x="CheckingStatus", y="Default Rate %",
            color="Default Rate %",
            color_continuous_scale=["#c8d9f5", "#0f52ba", "#e03131"],
            title="Default Rate by Checking Account Status", text="Default Rate %",
        )
        fig_c.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             textfont=dict(color="#1a1a2e"))
        _chart(fig_c, height=460, tickangle=-40)
        fig_c.update_layout(margin=dict(l=55, r=35, t=55, b=150))
        st.plotly_chart(fig_c, use_container_width=True)

    with col4:
        sav_risk = (
            df.groupby("ExistingSavings")["Default"]
            .mean().reset_index()
            .rename(columns={"Default": "Default Rate"})
        )
        sav_risk["Default Rate %"] = (sav_risk["Default Rate"] * 100).round(2)
        fig_s = px.bar(
            sav_risk.sort_values("Default Rate %", ascending=False),
            x="ExistingSavings", y="Default Rate %",
            color="Default Rate %",
            color_continuous_scale=["#c8d9f5", "#0f52ba", "#e03131"],
            title="Default Rate by Existing Savings Level", text="Default Rate %",
        )
        fig_s.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             textfont=dict(color="#1a1a2e"))
        _chart(fig_s, height=460, tickangle=-40)
        fig_s.update_layout(margin=dict(l=55, r=35, t=55, b=150))
        st.plotly_chart(fig_s, use_container_width=True)

    _section("👷 Employment Duration vs Risk")
    emp_cnt = df.groupby(["EmploymentDuration", "Risk"]).size().reset_index(name="Count")
    fig_emp = px.bar(
        emp_cnt, x="EmploymentDuration", y="Count",
        color="Risk", barmode="group",
        color_discrete_map=COLOR_MAP,
        title="Loan Count by Employment Duration & Risk",
        text="Count",
    )
    fig_emp.update_traces(textposition="outside", textfont=dict(color="#1a1a2e"))
    _chart(fig_emp, height=400, tickangle=-30)
    st.plotly_chart(fig_emp, use_container_width=True)

    _section("🫧 3D Scatter — Amount × Duration × Age by Risk")
    _samp2 = df.sample(min(700, len(df)), random_state=3)
    fig_sc3d = go.Figure()
    for risk_val, color in [("No Risk", PALETTE["blue"]), ("Risk", PALETTE["red"])]:
        _sub2 = _samp2[_samp2["Risk"] == risk_val]
        fig_sc3d.add_trace(go.Scatter3d(
            x=_sub2["LoanDuration"],
            y=_sub2["LoanAmount_INR"],
            z=_sub2["Age"],
            mode="markers",
            name=risk_val,
            marker=dict(
                size=_sub2["Age"] / 10,          # size encodes age magnitude
                color=color,
                opacity=0.70,
                line=dict(width=0.3, color="#ffffff"),
            ),
            hovertemplate=(
                "<b>" + risk_val + "</b><br>"
                "Duration: %{x} mo<br>"
                "Amount: ₹%{y:,.0f}<br>"
                "Age: %{z} yrs<extra></extra>"
            ),
        ))
    fig_sc3d.update_layout(
        paper_bgcolor=_BG,
        scene=dict(
            xaxis=dict(title="Duration (mo)",   tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            yaxis=dict(title="Loan Amount (₹)", tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            zaxis=dict(title="Age (yrs)",       tickfont=dict(size=9, color=_TEXT),
                       backgroundcolor="#0a1628", gridcolor=_GRID, showbackground=True),
            camera=dict(eye=dict(x=1.6, y=-1.4, z=1.1)),
        ),
        title=dict(text="Duration × Amount × Age — 3D Bubble (drag to rotate)",
                   font=dict(size=14, color="#00d4aa")),
        legend=dict(bgcolor=_BG, bordercolor=_GRID, borderwidth=1,
                    font=dict(color=_TEXT)),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(color=_TEXT),
    )
    st.plotly_chart(fig_sc3d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER RISK PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Customer Risk Predictor":

    _section("📝 Enter Borrower Attributes")
    st.markdown(
        "<p style='color:#5a6a8a;font-size:0.88rem;margin-bottom:0.8rem;'>"
        "Fill in the applicant details below and click <b>Predict Default Risk</b>.</p>",
        unsafe_allow_html=True,
    )

    def _opts(col):
        return sorted(df[col].dropna().unique().tolist())

    with st.form("prediction_form"):

        # ── Row 1 ──
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        checking_status = r1c1.selectbox("Checking Status",  _opts("CheckingStatus"))
        loan_duration   = r1c2.slider("Loan Duration (mo)",
                                       int(df["LoanDuration"].min()),
                                       int(df["LoanDuration"].max()), 24)
        credit_history  = r1c3.selectbox("Credit History",   _opts("CreditHistory"))
        loan_purpose    = r1c4.selectbox("Loan Purpose",      _opts("LoanPurpose"))

        # ── Row 2 ──
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        loan_amount      = r2c1.number_input("Loan Amount (₹)",
                                              min_value=5_000, max_value=2_000_000,
                                              value=200_000, step=5_000)
        existing_savings = r2c2.selectbox("Existing Savings", _opts("ExistingSavings"))
        employment_dur   = r2c3.selectbox("Employment Duration", _opts("EmploymentDuration"))
        installment_pct  = r2c4.slider("Installment % of Income", 1, 4, 3)

        # ── Row 3 ──
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        sex            = r3c1.selectbox("Sex",              _opts("Sex"))
        others_on_loan = r3c2.selectbox("Others on Loan",   _opts("OthersOnLoan"))
        residence_dur  = r3c3.slider("Residence Duration",  1, 4, 3)
        owns_property  = r3c4.selectbox("Owns Property",    _opts("OwnsProperty"))

        # ── Row 4 ──
        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        age               = r4c1.slider("Age (years)", 18, 80, 35)
        installment_plans = r4c2.selectbox("Installment Plans", _opts("InstallmentPlans"))
        housing           = r4c3.selectbox("Housing",           _opts("Housing"))
        credits_count     = r4c4.slider("Existing Credits Count", 1, 4, 1)

        # ── Row 5 ──
        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        job            = r5c1.selectbox("Job Category",   _opts("Job"))
        dependents     = r5c2.slider("Dependents",        1, 2, 1)
        telephone      = r5c3.selectbox("Telephone",      _opts("Telephone"))
        foreign_worker = r5c4.selectbox("Foreign Worker", _opts("ForeignWorker"))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Predict Default Risk",
                                          use_container_width=True)

    if submitted:
        input_dict = {
            "CheckingStatus":           checking_status,
            "LoanDuration":             loan_duration,
            "CreditHistory":            credit_history,
            "LoanPurpose":              loan_purpose,
            "LoanAmount_INR":           loan_amount,
            "ExistingSavings":          existing_savings,
            "EmploymentDuration":       employment_dur,
            "InstallmentPercent":       installment_pct,
            "Sex":                      sex,
            "OthersOnLoan":             others_on_loan,
            "CurrentResidenceDuration": residence_dur,
            "OwnsProperty":             owns_property,
            "Age":                      age,
            "InstallmentPlans":         installment_plans,
            "Housing":                  housing,
            "ExistingCreditsCount":     credits_count,
            "Job":                      job,
            "Dependents":               dependents,
            "Telephone":                telephone,
            "ForeignWorker":            foreign_worker,
        }

        prob = predict_risk(model, label_encoders, input_dict)
        pct  = prob * 100
        THRESHOLD = 0.35

        _section("📋 Prediction Result")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            bar_color = PALETTE["red"] if prob > THRESHOLD else PALETTE["blue"]
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                # number moved above gauge arc via domain — avoids overlapping the needle
                number={"suffix": "%", "font": {"size": 36, "color": "#1a1a2e"}},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                              "tickcolor": "#5a6a8a", "tickfont": {"color": "#1a1a2e", "size": 10}},
                    "bar":  {"color": bar_color, "thickness": 0.22},
                    "bgcolor": "#f8f9fb",
                    "borderwidth": 2,
                    "bordercolor": "#dde3ef",
                    "steps": [
                        {"range": [0,  35], "color": "#d0e8ff"},
                        {"range": [35, 65], "color": "#ffe4b0"},
                        {"range": [65, 100], "color": "#ffd0d0"},
                    ],
                    "threshold": {
                        "line": {"color": PALETTE["red"], "width": 3},
                        "thickness": 0.85,
                        "value": THRESHOLD * 100,
                    },
                },
                title={"text": "Default Probability",
                       "font": {"size": 14, "color": "#00d4aa"}},
            ))
            gauge.update_layout(
                paper_bgcolor=_BG,
                font={"color": _TEXT, "family": "Segoe UI, sans-serif"},
                height=320,
                # tall bottom margin pushes number display up above the arc
                margin=dict(l=30, r=30, t=60, b=80),
            )
            st.plotly_chart(gauge, use_container_width=True)

        with res_col2:
            if prob > THRESHOLD:
                st.error(
                    f"### 🚨 HIGH DEFAULT RISK — {pct:.1f}%\n\n"
                    "This applicant exhibits elevated credit risk characteristics. "
                    "The model predicts a default probability **above the 35% underwriting threshold**."
                )
                st.info(
                    "**📋 Underwriter Action Required:**\n\n"
                    "- Request additional collateral or a co-applicant guarantee.\n"
                    "- Reduce requested loan amount by at least 20%.\n"
                    "- Shorten loan tenure to ≤ 24 months.\n"
                    "- Escalate to Senior Credit Officer for manual review.\n"
                    "- Verify employment and income documents independently."
                )
            else:
                st.success(
                    f"### ✅ LOW DEFAULT RISK — {pct:.1f}%\n\n"
                    "This applicant meets the credit quality threshold. "
                    "The model predicts a default probability **below the 35% underwriting threshold**."
                )
                st.info(
                    "**📋 Underwriter Recommendation:**\n\n"
                    "- Standard loan approval process may proceed.\n"
                    "- Offer preferential interest rate tier (Prime – 0.5%).\n"
                    "- Flag as a cross-sell opportunity for insurance or savings products.\n"
                    "- Review again at the 12-month mark for credit limit upgrade."
                )

        _section("🔑 Top Feature Importances (Model-Wide)")
        fi = metrics["feature_importance"].head(12).reset_index()
        fi.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#003d2e"], [1, "#00d4aa"]],
            title="Top 12 Predictive Features", text="Importance",
        )
        fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                              textfont=dict(color=_TEXT))
        fig_fi.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_PLOT,
            height=430,
            margin=dict(l=165, r=70, t=50, b=30),
            yaxis=dict(autorange="reversed", gridcolor=_GRID, tickfont=dict(color=_TEXT)),
            xaxis=dict(gridcolor=_GRID, tickfont=dict(color=_TEXT)),
            font=dict(size=12, color=_TEXT),
            coloraxis_showscale=False,
            title_font=dict(size=14, color="#00d4aa"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":

    _section("📈 Model Scorecard")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Accuracy",               f"{metrics['accuracy']:.2%}")
    mc2.metric("AUC-ROC",                f"{metrics['roc_auc']:.4f}")
    rpt = metrics["report"]
    mc3.metric("F1 Score (weighted)",    f"{rpt['weighted avg']['f1-score']:.4f}")

    _section("📉 Confusion Matrix & Classification Report")
    col_cm, col_cr = st.columns(2)

    with col_cm:
        cm     = metrics["cm"]
        labels = ["No Risk", "Risk"]
        # Build a true 3D bar chart using Mesh3d cuboids — one per cell
        # Each bar is an axis-aligned box; count label is kept inside z-range
        _bar_colors = ["#0f52ba", "#e03131", "#e03131", "#0f52ba"]  # TN, FP, FN, TP
        fig_cm = go.Figure()
        _positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, (row_i, col_j) in enumerate(_positions):
            val = int(cm[row_i, col_j])
            x0, x1 = col_j + 0.05, col_j + 0.9
            y0, y1 = row_i + 0.05, row_i + 0.9
            z0, z1 = 0, val
            # 8 vertices of a cuboid
            fig_cm.add_trace(go.Mesh3d(
                x=[x0,x1,x1,x0,x0,x1,x1,x0],
                y=[y0,y0,y1,y1,y0,y0,y1,y1],
                z=[z0,z0,z0,z0,z1,z1,z1,z1],
                i=[0,0,0,0,4,4,2,2],
                j=[1,2,3,5,5,6,3,6],
                k=[2,3,7,4,6,7,7,7],
                color=_bar_colors[idx],
                opacity=0.85,
                flatshading=True,
                showlegend=False,
                hovertemplate=f"<b>{labels[row_i]} → {labels[col_j]}</b><br>Count: {val}<extra></extra>",
            ))
            # Label sitting just inside the top face — well within z range
            fig_cm.add_trace(go.Scatter3d(
                x=[(x0+x1)/2], y=[(y0+y1)/2], z=[max(val * 0.5, 5)],
                mode="text",
                text=[str(val)],
                textfont=dict(size=18, color="#ffffff"),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig_cm.update_layout(
            paper_bgcolor="#1a1a2e",
            scene=dict(
                xaxis=dict(
                    title="Predicted",
                    tickvals=[0.5, 1.5], ticktext=labels,
                    tickfont=dict(size=10, color="#e0e8ff"),
                    title_font=dict(color="#e0e8ff"),
                    backgroundcolor="#0d1b2a", gridcolor="#2a3f5f",
                    showbackground=True,
                ),
                yaxis=dict(
                    title="Actual",
                    tickvals=[0.5, 1.5], ticktext=labels,
                    tickfont=dict(size=10, color="#e0e8ff"),
                    title_font=dict(color="#e0e8ff"),
                    backgroundcolor="#0d1b2a", gridcolor="#2a3f5f",
                    showbackground=True,
                ),
                zaxis=dict(
                    title="Count",
                    tickfont=dict(size=10, color="#e0e8ff"),
                    title_font=dict(color="#e0e8ff"),
                    backgroundcolor="#0d1b2a", gridcolor="#2a3f5f",
                    showbackground=True,
                ),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.4)),
            ),
            title=dict(text="Confusion Matrix — 3D Bars (drag to rotate)",
                       font=dict(size=14, color="#e0e8ff")),
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
            font=dict(color="#e0e8ff"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cr:
        cls_df = pd.DataFrame(rpt).T.drop(["accuracy", "macro avg", "weighted avg"])
        cls_df = cls_df.rename(index={"0": "No Risk", "1": "Risk"}).reset_index()
        cls_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        fig_cr = px.bar(
            cls_df.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1-Score"]),
            x="variable", y="value", color="Class", barmode="group",
            color_discrete_map=COLOR_MAP,
            text="value",
            title="Classification Report Metrics",
            labels={"variable": "Metric", "value": "Score"},
        )
        fig_cr.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                              textfont=dict(color="#1a1a2e"))
        _chart(fig_cr, height=380, tickangle=0)
        st.plotly_chart(fig_cr, use_container_width=True)

    _section("📊 Feature Importances — All Features")
    fi_all = metrics["feature_importance"].reset_index()
    fi_all.columns = ["Feature", "Importance"]
    fig_fi_all = px.bar(
        fi_all, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#003d2e"], [1, "#00d4aa"]],
        title="Feature Importance — Gradient Boosting Model",
        text="Importance",
    )
    fig_fi_all.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                              textfont=dict(color=_TEXT))
    fig_fi_all.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_PLOT,
        height=540,
        margin=dict(l=185, r=70, t=55, b=30),
        yaxis=dict(autorange="reversed", gridcolor=_GRID, tickfont=dict(color=_TEXT)),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(color=_TEXT)),
        font=dict(size=12, color=_TEXT),
        coloraxis_showscale=False,
        title_font=dict(size=14, color="#00d4aa"),
    )
    st.plotly_chart(fig_fi_all, use_container_width=True)
