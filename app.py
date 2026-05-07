import streamlit as st
import numpy as np
import pandas as pd
import re, ast, warnings
from collections import Counter
warnings.filterwarnings("ignore")

# ── sklearn imports ──────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Role Predictor",
    page_icon="🎯",
    layout="wide"
)

# ── minimal custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }

    /* Top banner */
    .banner {
        background: #1e3a5f;
        border-radius: 10px;
        padding: 22px 30px 16px;
        margin-bottom: 28px;
        color: white;
    }
    .banner h1 { font-size: 1.9rem; margin: 0 0 4px; font-weight: 700; }
    .banner p  { font-size: 0.92rem; margin: 0; opacity: .8; }

    /* Metric cards */
    .metric-card {
        background: #f0f4ff;
        border: 1px solid #c8d8f5;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-card .label { font-size: 0.78rem; color: #5a6a85; text-transform: uppercase; letter-spacing: .05em; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #1e3a5f; }

    /* Result rows */
    .result-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 8px;
        background: #f8faff;
        border-left: 4px solid #3a7bd5;
    }
    .result-rank { font-weight: 700; font-size: 1.1rem; color: #1e3a5f; min-width: 22px; }
    .result-role { flex: 1; font-weight: 600; color: #2d2d2d; }
    .result-bar  { height: 8px; border-radius: 4px; background: #3a7bd5; }

    /* Section headings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e3a5f;
        border-bottom: 2px solid #e0e8ff;
        padding-bottom: 6px;
        margin: 20px 0 14px;
    }

    /* Note box */
    .note-box {
        background: #fffbea;
        border: 1px solid #f0d060;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #6b5700;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_train():
    """Load dataset, train all three models, return everything needed."""

    # ── load ─────────────────────────────────────────────────────────────────
    df_raw = pd.read_csv("final_dataset.csv", index_col=0)

    # ── parse skills ─────────────────────────────────────────────────────────
    def parse_skills(raw):
        if pd.isna(raw):
            return ""
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return " ".join(str(s).strip().lower() for s in parsed)
        except Exception:
            pass
        return " ".join(re.sub(r"[()'\",]", " ", str(raw)).split())

    df = df_raw.copy()
    df["skills_text"] = df["skills_clean"].apply(parse_skills)
    df["education"]   = df["education"].fillna("Unknown")
    df["experience_years"] = df["experience_years"].fillna(df["experience_years"].median())

    # ── filter rare roles ────────────────────────────────────────────────────
    counts = df["job_role"].value_counts()
    valid  = counts[counts >= 20].index
    df     = df[df["job_role"].isin(valid)].reset_index(drop=True)

    # ── label encoding ───────────────────────────────────────────────────────
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["job_role"])

    # ── features ─────────────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000,
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

    # ── train ─────────────────────────────────────────────────────────────────
    lr  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_neighbors=7, metric="cosine",
                               algorithm="brute", n_jobs=-1)
    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)

    lr.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    preds = {
        "Logistic Regression": lr.predict(X_test),
        "KNN (k=7, cosine)":   knn.predict(X_test),
        "LinearSVC":           svm.predict(X_test),
    }

    def metrics(y_true, y_pred, name):
        return {
            "Model":         name,
            "Accuracy":      accuracy_score(y_true, y_pred),
            "Precision (W)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall (W)":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1 (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1 (Macro)":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    results = pd.DataFrame([
        metrics(y_test, preds["Logistic Regression"], "Logistic Regression"),
        metrics(y_test, preds["KNN (k=7, cosine)"],   "KNN (k=7, cosine)"),
        metrics(y_test, preds["LinearSVC"],            "LinearSVC"),
    ]).set_index("Model").round(4)

    # ── skill frequency ───────────────────────────────────────────────────────
    all_raw = []
    for raw in df["skills_clean"]:
        try:
            parsed = ast.literal_eval(raw)
            all_raw.extend(s.strip().lower() for s in parsed)
        except Exception:
            pass
    skill_freq = Counter(all_raw)

    return {
        "df": df, "le": le, "tfidf": tfidf, "ohe": ohe, "scaler": scaler,
        "lr": lr, "knn": knn, "svm": svm,
        "X_test": X_test, "y_test": y_test, "preds": preds,
        "results": results, "skill_freq": skill_freq,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🎯 Skill-Based Job Role Predictor</h1>
  <p>Supervised ML Classification &nbsp;|&nbsp; Logistic Regression · KNN · LinearSVC &nbsp;|&nbsp; 
     TF-IDF + One-Hot + Scaled Experience</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA — with spinner
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Training models on the dataset — hang on a sec..."):
    try:
        data = load_and_train()
        load_ok = True
    except FileNotFoundError:
        load_ok = False

if not load_ok:
    st.error("⚠️ `final_dataset.csv` not found. Place it in the same folder as `app.py` and restart.")
    st.info("The app expects the dataset used in the notebook: **final_dataset.csv**")
    st.stop()

df         = data["df"]
le         = data["le"]
tfidf      = data["tfidf"]
ohe        = data["ohe"]
scaler     = data["scaler"]
lr_model   = data["lr"]
knn_model  = data["knn"]
svm_model  = data["svm"]
X_test     = data["X_test"]
y_test     = data["y_test"]
preds      = data["preds"]
results    = data["results"]
skill_freq = data["skill_freq"]

NUM_CLASSES = len(le.classes_)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Job Role",
    "📊 Model Results",
    "📈 EDA",
    "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Enter your profile</div>', unsafe_allow_html=True)

    col_form, col_out = st.columns([1, 1.1], gap="large")

    with col_form:
        skills_input = st.text_area(
            "Skills (comma-separated)",
            placeholder="e.g. python, machine learning, sql, pandas, statistics",
            height=110,
        )

        edu_options = sorted(df["education"].unique().tolist())
        education   = st.selectbox("Highest Education", edu_options)

        experience  = st.slider("Years of Experience", 0.0, 30.0, 2.0, 0.5)

        model_choice = st.radio(
            "Model to use",
            ["LinearSVC (best)", "Logistic Regression", "KNN (k=7, cosine)"],
            horizontal=True,
        )

        top_k = st.slider("Show top N predictions", 1, 10, 5)

        predict_btn = st.button("🔍 Predict", use_container_width=True)

    with col_out:
        if predict_btn:
            if not skills_input.strip():
                st.warning("Please enter at least one skill.")
            else:
                # ── build feature vector ──────────────────────────────────
                skills_list = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
                skills_str  = " ".join(skills_list)

                X_s  = tfidf.transform([skills_str])
                X_e  = ohe.transform([[education]])
                X_x  = csr_matrix(scaler.transform([[experience]]))
                X_new = hstack([X_s, X_e, X_x])

                # ── choose model ──────────────────────────────────────────
                if model_choice.startswith("LinearSVC"):
                    scores   = svm_model.decision_function(X_new)[0]
                    top_ids  = np.argsort(scores)[::-1][:top_k]
                    top_vals = scores[top_ids]
                    score_label = "Decision score"
                elif model_choice.startswith("Logistic"):
                    probs    = lr_model.predict_proba(X_new)[0]
                    top_ids  = np.argsort(probs)[::-1][:top_k]
                    top_vals = probs[top_ids]
                    score_label = "Probability"
                else:
                    # KNN — use decision-function style from probabilities
                    probs    = knn_model.predict_proba(X_new)[0]
                    top_ids  = np.argsort(probs)[::-1][:top_k]
                    top_vals = probs[top_ids]
                    score_label = "Probability"

                st.markdown(f'<div class="section-title">Top {top_k} Predicted Roles</div>',
                            unsafe_allow_html=True)

                # Normalise bar widths to [0,1] relative to top score
                max_val = max(top_vals) if max(top_vals) != 0 else 1

                for rank, (idx, val) in enumerate(zip(top_ids, top_vals), 1):
                    role    = le.classes_[idx]
                    bar_pct = int((val / max_val) * 100)
                    label   = f"{val:.1%}" if score_label == "Probability" else f"{val:.2f}"

                    color = "#3a7bd5" if rank == 1 else "#6ea8de" if rank <= 3 else "#a8caf0"
                    st.markdown(f"""
                    <div class="result-row" style="border-left-color:{color};">
                      <div class="result-rank">#{rank}</div>
                      <div class="result-role">{role}</div>
                      <div style="display:flex;align-items:center;gap:8px;flex:1;">
                        <div class="result-bar" style="width:{bar_pct}%;background:{color};"></div>
                        <span style="font-size:.8rem;color:#555;white-space:nowrap;">{label}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="note-box">
                  Skills entered: <strong>{', '.join(skills_list)}</strong><br>
                  Model: <strong>{model_choice}</strong> &nbsp;|&nbsp;
                  Score type: <strong>{score_label}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Fill in your profile and click **Predict** to see job role matches.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    # ── metric cards ─────────────────────────────────────────────────────────
    best_model = results["Accuracy"].idxmax()
    best_acc   = results.loc[best_model, "Accuracy"]
    best_f1    = results.loc[best_model, "F1 (Weighted)"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Dataset size</div>
          <div class="value">{len(df):,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Job role classes</div>
          <div class="value">{NUM_CLASSES}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Best accuracy</div>
          <div class="value">{best_acc:.1%}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Best F1 (W)</div>
          <div class="value">{best_f1:.1%}</div></div>""", unsafe_allow_html=True)

    st.write("")

    # ── results table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(
        results.style.highlight_max(axis=0, color="#d4e9ff").format("{:.4f}"),
        use_container_width=True,
    )

    # ── bar chart ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)

    metrics_cols = ["Accuracy", "Precision (W)", "Recall (W)", "F1 (Weighted)", "F1 (Macro)"]
    colors_bar   = ["#3a7bd5", "#e8703a", "#3db87a"]
    x = np.arange(len(metrics_cols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, row) in enumerate(results.iterrows()):
        bars = ax.bar(x + i * width, row[metrics_cols], width,
                      label=name, color=colors_bar[i], edgecolor="white")
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.004, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_cols, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── confusion matrix (SVM, top 12 roles) ─────────────────────────────────
    st.markdown('<div class="section-title">Confusion Matrix — LinearSVC (Top 12 Roles)</div>',
                unsafe_allow_html=True)

    top_ids_cm   = np.argsort(np.bincount(y_test))[-12:]
    mask         = np.isin(y_test, top_ids_cm)
    y_t_top      = y_test[mask]
    y_p_top      = preds["LinearSVC"][mask]
    top_labels   = le.classes_[top_ids_cm]
    cm           = confusion_matrix(y_t_top, y_p_top, labels=top_ids_cm)

    fig2, ax2 = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=top_labels, yticklabels=top_labels,
                linewidths=0.4, linecolor="lightgray", ax=ax2)
    ax2.set_title("Confusion Matrix — LinearSVC (Top 12 Roles)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Predicted", fontsize=10)
    ax2.set_ylabel("Actual", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Experience distribution
    with c1:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(df["experience_years"], bins=14, color="#3a7bd5",
                 edgecolor="white", linewidth=0.8)
        ax3.set_title("Distribution of Experience Years", fontweight="bold", fontsize=11)
        ax3.set_xlabel("Years of Experience")
        ax3.set_ylabel("Count")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # Top education
    with c2:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        top_edu = df["education"].value_counts().head(12)
        ax4.barh(top_edu.index[::-1], top_edu.values[::-1], color="#e8703a", edgecolor="white")
        ax4.set_title("Top 12 Education Backgrounds", fontweight="bold", fontsize=11)
        ax4.set_xlabel("Count")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # Top 20 job roles
    st.markdown('<div class="section-title">Top 20 Job Roles</div>', unsafe_allow_html=True)
    fig5, ax5 = plt.subplots(figsize=(11, 6))
    top_roles = df["job_role"].value_counts().head(20)
    palette   = sns.color_palette("Blues_r", 20)
    ax5.barh(top_roles.index[::-1], top_roles.values[::-1], color=palette, edgecolor="white")
    ax5.set_title("Top 20 Most Frequent Job Roles", fontweight="bold", fontsize=12)
    ax5.set_xlabel("Number of Listings")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # Top 30 skills
    st.markdown('<div class="section-title">Top 30 Most Common Skills</div>', unsafe_allow_html=True)
    top_skills = pd.Series(dict(skill_freq.most_common(30)))
    fig6, ax6  = plt.subplots(figsize=(11, 7))
    top_skills[::-1].plot(kind="barh", ax=ax6, color="#3db87a", edgecolor="white")
    ax6.set_title("Top 30 Skills Across All Job Roles", fontweight="bold", fontsize=12)
    ax6.set_xlabel("Frequency")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Goal:** Predict a person's most likely job role based on their skills, education level,
and years of experience using supervised machine learning.

**Dataset:** `final_dataset.csv` — {len(df):,} samples after filtering roles with fewer than 20 samples,
resulting in **{NUM_CLASSES} unique job role classes**.

**Feature Engineering:**
- **TF-IDF (skills text)** — captures how distinctive each skill is across all roles  
  (ngram_range 1–2, max 3000 features, sublinear TF normalization)
- **One-Hot Encoding (education)** — categorical encoding of education level
- **Scaled experience years** — StandardScaler for numerical stability

**Models trained:**

| Model | Notes |
|---|---|
| Logistic Regression | Reliable linear baseline; returns calibrated probabilities |
| KNN (k=7, cosine) | Non-parametric; cosine metric suits high-dimensional sparse data |
| LinearSVC | Best performer on sparse TF-IDF; fast training, no probability output |

**Key findings:**
- LinearSVC achieves the highest accuracy on this dataset
- Skills text (TF-IDF) is by far the most informative feature
- Education and experience add marginal but consistent improvements
- Multi-class classification with 300+ classes requires weighted F1 evaluation
    """)

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
1. You enter a list of skills (comma-separated), your education level, and years of experience  
2. Skills are TF-IDF vectorised using the same vocabulary fitted on training data  
3. Education is one-hot encoded; experience is scaled to match training distribution  
4. The selected model scores each possible job role  
5. The top N roles with the highest scores are returned
    """)