import warnings
warnings.filterwarnings("ignore")

import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Role Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #101820;
    border-right: 1px solid #1c2b3a;
}
[data-testid="stSidebar"] * { color: #c0d0e0 !important; }
[data-testid="stSidebar"] label {
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6a8aaa !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #6a8aaa !important;
    font-size: 0.76rem;
}
[data-testid="stSidebar"] hr { border-color: #1c2b3a !important; }

.main .block-container {
    background: #f4f6f9;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

.page-header {
    background: #101820;
    border-radius: 6px;
    padding: 22px 28px 18px;
    margin-bottom: 20px;
    border-left: 4px solid #2f80ed;
}
.page-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #fff;
    margin: 0 0 5px;
    letter-spacing: -0.01em;
}
.page-header p { font-size: 0.8rem; color: #6a8aaa; margin: 0; line-height: 1.5; }

.stats-row { display: flex; gap: 12px; margin-bottom: 18px; }
.stat-card {
    flex: 1; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: 14px 16px 12px;
}
.stat-card.c1 { border-top: 3px solid #2f80ed; }
.stat-card.c2 { border-top: 3px solid #f97316; }
.stat-card.c3 { border-top: 3px solid #10b981; }
.stat-card.c4 { border-top: 3px solid #8b5cf6; }
.stat-label { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.65rem; font-weight: 600; color: #101820; line-height: 1.1; margin: 2px 0; }
.stat-sub   { font-size: 0.7rem; color: #94a3b8; }

.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; font-weight: 600; color: #2f80ed;
    text-transform: uppercase; letter-spacing: 0.12em;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px; margin: 18px 0 12px;
}

.chart-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: 16px 16px 10px; margin-bottom: 12px;
}
.chart-card-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 600;
}

.pred-wrap { margin-top: 4px; }
.pred-item {
    display: flex; align-items: center; gap: 12px;
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 5px; padding: 10px 14px; margin-bottom: 7px;
}
.pred-item.rank1 { border-left: 3px solid #2f80ed; }
.pred-item.rank2 { border-left: 3px solid #10b981; }
.pred-item.rank3 { border-left: 3px solid #f97316; }
.pred-item.rankN { border-left: 3px solid #cbd5e1; }
.pred-num  { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #94a3b8; min-width: 22px; }
.pred-role { flex: 1; font-weight: 600; font-size: 0.88rem; color: #101820; }
.pred-bar-bg { width: 90px; height: 5px; background: #e2e8f0; border-radius: 3px; }
.pred-bar    { height: 5px; border-radius: 3px; }
.pred-score  { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #2f80ed; min-width: 44px; text-align: right; }

.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 5px;
    padding: 10px 14px; font-size: 0.8rem; color: #1d4ed8; margin-top: 8px;
}
.warn-box {
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 5px;
    padding: 10px 14px; font-size: 0.8rem; color: #9a3412; margin-bottom: 10px;
}

.stButton > button {
    background: #101820; color: #fff; border: 1px solid #1c2b3a;
    border-radius: 4px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em;
    padding: 10px 20px; width: 100%;
}
.stButton > button:hover { background: #1c2b3a; border-color: #2f80ed; color: #fff; }

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; letter-spacing: 0.03em;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#f1f5f9",
    "grid.linewidth":    0.8,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#1e293b",
    "axes.labelcolor":   "#475569",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
})

BLUE   = "#2f80ed"
ORANGE = "#f97316"
GREEN  = "#10b981"
PURPLE = "#8b5cf6"
DARK   = "#101820"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_everything():
    df_raw = pd.read_csv("final_dataset.csv", index_col=0)

    def parse_skills(raw):
        if pd.isna(raw):
            return ""
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return " ".join(str(s).strip().lower() for s in parsed)
        except Exception:
            pass
        return " ".join(str(raw).split())

    df = df_raw.copy()
    df["skills_text"]      = df["skills_clean"].apply(parse_skills)
    df["education"]        = df["education"].fillna("Unknown")
    df["experience_years"] = df["experience_years"].fillna(df["experience_years"].median())

    counts = df["job_role"].value_counts()
    valid  = counts[counts >= 20].index
    df     = df[df["job_role"].isin(valid)].reset_index(drop=True)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["job_role"])

    tfidf    = TfidfVectorizer(ngram_range=(1, 2), max_features=3000,
                               sublinear_tf=True, min_df=2, strip_accents="unicode")
    X_skills = tfidf.fit_transform(df["skills_text"])

    ohe   = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_edu = ohe.fit_transform(df[["education"]])

    scaler = StandardScaler()
    X_exp  = csr_matrix(scaler.fit_transform(df[["experience_years"]]))

    X = hstack([X_skills, X_edu, X_exp])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr  = joblib.load("lr_model.joblib")
    knn = joblib.load("knn_model.joblib")
    svm = joblib.load("svm_model.joblib")

    model_preds = {
        "Logistic Regression": lr.predict(X_test),
        "KNN (k=7)":           knn.predict(X_test),
        "LinearSVC":           svm.predict(X_test),
    }

    def mets(yt, yp):
        return {
            "Accuracy":      round(accuracy_score(yt, yp), 4),
            "Precision (W)": round(precision_score(yt, yp, average="weighted", zero_division=0), 4),
            "Recall (W)":    round(recall_score(yt, yp, average="weighted", zero_division=0), 4),
            "F1 (Weighted)": round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
            "F1 (Macro)":    round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        }

    results = {k: mets(y_test, v) for k, v in model_preds.items()}

    all_skills = []
    for raw in df["skills_clean"]:
        try:
            parsed = ast.literal_eval(raw)
            all_skills.extend(s.strip().lower() for s in parsed)
        except Exception:
            pass
    skill_freq = Counter(all_skills)

    return dict(
        df=df, le=le, tfidf=tfidf, ohe=ohe, scaler=scaler,
        lr=lr, knn=knn, svm=svm,
        X_test=X_test, y_test=y_test,
        model_preds=model_preds, results=results, skill_freq=skill_freq,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Job Role Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Predict", "Dashboard", "Model Results"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Project**")
    st.markdown(
        "Supervised ML classification of job roles "
        "from skills, education, and years of experience."
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("11,167 combined records  \n324 unique job roles  \nSource: Kaggle")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("Logistic Regression  \nKNN (k=7, cosine)  \nLinearSVC")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading models and preprocessing data..."):
    try:
        data = load_everything()
        ok   = True
    except FileNotFoundError as e:
        ok  = False
        err = str(e)

if not ok:
    st.error(f"Could not load required files: {err}")
    st.info(
        "Place `final_dataset.csv`, `lr_model.joblib`, "
        "`knn_model.joblib`, `svm_model.joblib` in the same folder as `app.py`."
    )
    st.stop()

df          = data["df"]
le          = data["le"]
tfidf       = data["tfidf"]
ohe         = data["ohe"]
scaler      = data["scaler"]
lr_model    = data["lr"]
knn_model   = data["knn"]
svm_model   = data["svm"]
X_test      = data["X_test"]
y_test      = data["y_test"]
model_preds = data["model_preds"]
results     = data["results"]
skill_freq  = data["skill_freq"]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
SUBTITLES = {
    "Predict":       "Enter your profile and get matched job roles",
    "Dashboard":     "Exploratory analysis of the training dataset",
    "Model Results": "Evaluation metrics for all three classifiers",
}
st.markdown(f"""
<div class="page-header">
  <h1>Skill-Based Job Role Predictor</h1>
  <p>Machine Learning Course Project &nbsp;|&nbsp; {SUBTITLES[page]}</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════
if page == "Predict":

    col_in, col_out = st.columns([1, 1.15], gap="large")

    with col_in:
        st.markdown('<div class="sec-title">Your Profile</div>', unsafe_allow_html=True)

        skills_raw = st.text_area(
            "Skills",
            placeholder="e.g. python, machine learning, sql, communication, leadership",
            height=120,
            help="Enter your skills separated by commas",
        )
        edu_opts  = sorted(df["education"].unique().tolist())
        education = st.selectbox("Education level", edu_opts)
        experience = st.slider("Years of experience", 0.0, 13.0, 2.0, 0.5)
        model_name = st.radio(
            "Classifier",
            ["Logistic Regression", "KNN (k=7)", "LinearSVC"],
            horizontal=True,
        )
        top_k = st.slider("Number of results", 1, 10, 5)
        run   = st.button("Run prediction")

    with col_out:
        st.markdown('<div class="sec-title">Prediction Results</div>', unsafe_allow_html=True)

        if run:
            if not skills_raw.strip():
                st.markdown(
                    '<div class="warn-box">Please enter at least one skill.</div>',
                    unsafe_allow_html=True,
                )
            else:
                skills_list = [s.strip().lower() for s in skills_raw.split(",") if s.strip()]
                skills_str  = " ".join(skills_list)

                X_s   = tfidf.transform([skills_str])
                X_e   = ohe.transform([[education]])
                X_x   = csr_matrix(scaler.transform([[experience]]))
                X_new = hstack([X_s, X_e, X_x])

                model_map = {
                    "Logistic Regression": lr_model,
                    "KNN (k=7)":           knn_model,
                    "LinearSVC":           svm_model,
                }
                m = model_map[model_name]

                if model_name == "LinearSVC":
                    raw_sc      = m.decision_function(X_new)[0]
                    top_idx     = np.argsort(raw_sc)[::-1][:top_k]
                    top_vals    = raw_sc[top_idx]
                    score_label = "decision score"
                else:
                    probs       = m.predict_proba(X_new)[0]
                    top_idx     = np.argsort(probs)[::-1][:top_k]
                    top_vals    = probs[top_idx]
                    score_label = "probability"

                max_val   = max(abs(top_vals)) if max(abs(top_vals)) > 0 else 1
                rank_css  = ["rank1", "rank2", "rank3"] + ["rankN"] * 10
                bar_colors = [BLUE, GREEN, ORANGE, "#94a3b8"]

                st.markdown('<div class="pred-wrap">', unsafe_allow_html=True)
                for i, (idx, val) in enumerate(zip(top_idx, top_vals)):
                    role    = le.classes_[idx]
                    pct     = int(abs(val) / max_val * 100)
                    bc      = bar_colors[min(i, 3)]
                    disp    = f"{val:.1%}" if score_label == "probability" else f"{val:.2f}"
                    rc      = rank_css[i]
                    st.markdown(f"""
                    <div class="pred-item {rc}">
                      <span class="pred-num">#{i+1}</span>
                      <span class="pred-role">{role}</span>
                      <div class="pred-bar-bg">
                        <div class="pred-bar" style="width:{pct}%;background:{bc};"></div>
                      </div>
                      <span class="pred-score">{disp}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                  Model: <b>{model_name}</b> &nbsp;|&nbsp;
                  Skills: <b>{len(skills_list)}</b> &nbsp;|&nbsp;
                  Score type: <b>{score_label}</b>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box">Fill in your profile on the left '
                'and click <b>Run prediction</b> to see matching job roles.</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":

    # stat cards
    skill_counts_list = []
    for raw in df["skills_clean"]:
        try:
            skill_counts_list.append(len(ast.literal_eval(raw)))
        except Exception:
            skill_counts_list.append(0)

    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card c1">
        <div class="stat-label">Records</div>
        <div class="stat-value">{len(df):,}</div>
        <div class="stat-sub">after filtering rare roles</div>
      </div>
      <div class="stat-card c2">
        <div class="stat-label">Job roles</div>
        <div class="stat-value">{df['job_role'].nunique()}</div>
        <div class="stat-sub">min 20 samples each</div>
      </div>
      <div class="stat-card c3">
        <div class="stat-label">Unique skills</div>
        <div class="stat-value">{len(skill_freq):,}</div>
        <div class="stat-sub">across all records</div>
      </div>
      <div class="stat-card c4">
        <div class="stat-label">Median skills/record</div>
        <div class="stat-value">{int(np.median(skill_counts_list))}</div>
        <div class="stat-sub">skills per person</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # row 1: top roles + top skills
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Top 15 job roles by frequency</div>',
                    unsafe_allow_html=True)
        top_roles = df["job_role"].value_counts().head(15)
        fig, ax   = plt.subplots(figsize=(5.5, 4.8))
        colors_r  = [BLUE] + ["#c7d9f5"] * 14
        ax.barh(top_roles.index[::-1], top_roles.values[::-1],
                color=colors_r[::-1], edgecolor="white", linewidth=0.6, height=0.68)
        ax.set_xlabel("Number of records")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.tick_params(axis="y", labelsize=7.2)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Top 20 skills by frequency</div>',
                    unsafe_allow_html=True)
        top_skills = pd.Series(dict(skill_freq.most_common(20)))
        fig, ax    = plt.subplots(figsize=(5.5, 4.8))
        colors_s   = [ORANGE] + ["#fdd9bb"] * 19
        ax.barh(top_skills.index[::-1], top_skills.values[::-1],
                color=colors_s[::-1], edgecolor="white", linewidth=0.6, height=0.68)
        ax.set_xlabel("Frequency")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.tick_params(axis="y", labelsize=7.2)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # row 2: education + experience
    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Education level distribution</div>',
                    unsafe_allow_html=True)
        edu_map = {}
        for e in df["education"]:
            if "Bachelor" in e:              k = "Bachelor's"
            elif "Master" in e:              k = "Master's"
            elif "PhD" in e or "Doc" in e:   k = "PhD"
            elif "MBA" in e:                 k = "MBA"
            elif "High School" in e:         k = "High School"
            elif "Trade" in e:               k = "Trade School"
            elif "Associate" in e:           k = "Associate"
            else:                            k = "Other"
            edu_map[k] = edu_map.get(k, 0) + 1
        edu_s   = pd.Series(edu_map).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        wedge_colors = [BLUE, GREEN, ORANGE, PURPLE, "#94a3b8", "#f43f5e", "#0ea5e9", "#eab308"]
        wedges, texts, autotexts = ax.pie(
            edu_s.values, labels=edu_s.index,
            colors=wedge_colors[:len(edu_s)],
            autopct="%1.1f%%", startangle=140, pctdistance=0.78,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
            textprops=dict(fontsize=7.5),
        )
        for at in autotexts:
            at.set(fontsize=7, color="white", fontweight="bold")
        ax.set_aspect("equal")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Experience years distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df["experience_years"].dropna(), bins=14,
                color=GREEN, edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Years of experience")
        ax.set_ylabel("Count")
        med = df["experience_years"].median()
        ax.axvline(med, color=DARK, linewidth=1.3, linestyle="--")
        ax.text(med + 0.15, ax.get_ylim()[1] * 0.88,
                f"median = {med:.1f}", fontsize=7.5, color=DARK)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # row 3: skills per record + class size distribution
    c5, c6 = st.columns(2, gap="medium")

    with c5:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Number of skills per record</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.hist(skill_counts_list, bins=20, color=PURPLE,
                edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Skills per record")
        ax.set_ylabel("Count")
        med2 = float(np.median(skill_counts_list))
        ax.axvline(med2, color=DARK, linewidth=1.3, linestyle="--")
        ax.text(med2 + 0.2, ax.get_ylim()[1] * 0.88,
                f"median = {med2:.0f}", fontsize=7.5, color=DARK)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Records per job role (class size distribution)</div>',
                    unsafe_allow_html=True)
        role_counts = df["job_role"].value_counts().values
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.hist(role_counts, bins=25, color=ORANGE,
                edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Records per role")
        ax.set_ylabel("Number of roles")
        med3 = float(np.median(role_counts))
        ax.axvline(med3, color=DARK, linewidth=1.3, linestyle="--")
        ax.text(med3 + 0.3, ax.get_ylim()[1] * 0.88,
                f"median = {med3:.0f}", fontsize=7.5, color=DARK)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Results":

    best_name = max(results, key=lambda k: results[k]["Accuracy"])
    best_acc  = results[best_name]["Accuracy"]
    best_f1   = results[best_name]["F1 (Weighted)"]

    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card c1">
        <div class="stat-label">Best accuracy</div>
        <div class="stat-value">{best_acc:.1%}</div>
        <div class="stat-sub">{best_name}</div>
      </div>
      <div class="stat-card c2">
        <div class="stat-label">Best F1 weighted</div>
        <div class="stat-value">{best_f1:.1%}</div>
        <div class="stat-sub">{best_name}</div>
      </div>
      <div class="stat-card c3">
        <div class="stat-label">Test set size</div>
        <div class="stat-value">{len(y_test):,}</div>
        <div class="stat-sub">20% stratified split</div>
      </div>
      <div class="stat-card c4">
        <div class="stat-label">Classes evaluated</div>
        <div class="stat-value">324</div>
        <div class="stat-sub">min 20 samples each</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # metrics table
    st.markdown('<div class="sec-title">Metrics table</div>', unsafe_allow_html=True)
    rows = [
        {"Model": name, **m} for name, m in results.items()
    ]
    tbl = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        tbl.style
           .highlight_max(axis=0, color="#dbeafe")
           .format("{:.4f}"),
        use_container_width=True,
    )

    # charts
    ca, cb = st.columns(2, gap="medium")
    model_names = list(results.keys())
    colors_m    = [BLUE, ORANGE, GREEN]

    with ca:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Accuracy and F1 (weighted) by model</div>',
                    unsafe_allow_html=True)
        accs = [results[n]["Accuracy"]      for n in model_names]
        f1ws = [results[n]["F1 (Weighted)"] for n in model_names]
        x    = np.arange(len(model_names))
        w    = 0.35
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        b1 = ax.bar(x - w/2, accs, w, label="Accuracy",
                    color=BLUE,  edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x + w/2, f1ws, w, label="F1 (weighted)",
                    color=GREEN, edgecolor="white", linewidth=0.8)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_ylim(0, 0.80)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8, frameon=False)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">All five metrics by model</div>',
                    unsafe_allow_html=True)
        metric_keys = ["Accuracy", "Precision (W)", "Recall (W)", "F1 (Weighted)", "F1 (Macro)"]
        short_keys  = ["Acc", "Prec", "Rec", "F1-W", "F1-M"]
        x2  = np.arange(len(metric_keys))
        w3  = 0.22
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for i, (name, col) in enumerate(zip(model_names, colors_m)):
            vals = [results[name][mk] for mk in metric_keys]
            ax.bar(x2 + i * w3, vals, w3, label=name,
                   color=col, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x2 + w3)
        ax.set_xticklabels(short_keys, fontsize=8)
        ax.set_ylim(0, 0.80)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=7.5, frameon=False)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # confusion matrix selector
    st.markdown('<div class="sec-title">Confusion matrix — top 15 roles</div>',
                unsafe_allow_html=True)
    cm_model = st.selectbox("Select model", model_names, key="cm_select")

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Rows = actual label, Columns = predicted label. '
        'Only top 15 most frequent test roles shown.</div>',
        unsafe_allow_html=True,
    )
    top15_ids  = np.argsort(np.bincount(y_test))[-15:]
    mask       = np.isin(y_test, top15_ids)
    y_t15      = y_test[mask]
    y_p15      = model_preds[cm_model][mask]
    top15_lbl  = le.classes_[top15_ids]
    cm_mat     = confusion_matrix(y_t15, y_p15, labels=top15_ids)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=top15_lbl, yticklabels=top15_lbl,
        linewidths=0.4, linecolor="#f1f5f9",
        annot_kws={"size": 7.5}, ax=ax2,
    )
    ax2.set_xlabel("Predicted", fontsize=8.5)
    ax2.set_ylabel("Actual", fontsize=8.5)
    plt.xticks(rotation=38, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    # hyperparameters
    st.markdown('<div class="sec-title">Model hyperparameters</div>',
                unsafe_allow_html=True)
    params_df = pd.DataFrame([
        {"Model": "Logistic Regression",
         "Key parameters": "C=1.0, solver=lbfgs, max_iter=1000"},
        {"Model": "KNN (k=7)",
         "Key parameters": "n_neighbors=7, metric=cosine, algorithm=brute"},
        {"Model": "LinearSVC",
         "Key parameters": "C=1.0, max_iter=2000, random_state=42"},
    ]).set_index("Model")
    st.dataframe(params_df, use_container_width=True)
