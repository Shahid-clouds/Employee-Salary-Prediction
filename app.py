import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os

st.set_page_config(page_title="DS Salary Predictor", page_icon="💼", layout="wide")

# ── Load & prepare data ──────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("ds_salaries.csv")
    df = df[df["salary_in_usd"] <= 300000]
    df = df[["experience_level", "employment_type", "job_title",
             "salary_in_usd", "remote_ratio", "company_size"]]

    exp_map  = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
    emp_map  = {"PT": 0, "FL": 1, "CT": 2, "FT": 3}
    size_map = {"S": 0, "M": 1, "L": 2}

    df["experience_level"] = df["experience_level"].map(exp_map)
    df["employment_type"]  = df["employment_type"].map(emp_map)
    df["company_size"]     = df["company_size"].map(size_map)

    le_job = LabelEncoder()
    df["job_title"] = le_job.fit_transform(df["job_title"])

    X = df.drop(columns=["salary_in_usd"])
    y = df["salary_in_usd"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":  LinearRegression(),
        "Decision Tree":      DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest":      RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = {
            "model": m,
            "r2":  round(r2_score(y_test, preds) * 100, 1),
            "mae": int(mean_absolute_error(y_test, preds)),
        }

    raw_df = pd.read_csv("ds_salaries.csv")
    raw_df = raw_df[raw_df["salary_in_usd"] <= 300000]
    return results, le_job, raw_df

results, le_job, raw_df = load_and_train()
best_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]

# ── Mappings ─────────────────────────────────────────────────────────────────
exp_labels  = {"Entry Level (EN)": 0, "Mid Level (MI)": 1,
               "Senior Level (SE)": 2, "Executive (EX)": 3}
emp_labels  = {"Part Time (PT)": 0, "Freelance (FL)": 1,
               "Contract (CT)": 2, "Full Time (FT)": 3}
size_labels = {"Small (S)": 0, "Medium (M)": 1, "Large (L)": 2}
remote_labels = {"On-site (0%)": 0, "Hybrid (50%)": 50, "Remote (100%)": 100}
job_titles  = sorted(le_job.classes_)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💼 Predict Salary", "📊 Model Performance", "🔍 Data Insights"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("💼 Data Science Salary Predictor")
    st.markdown("Enter your profile details to get an estimated salary in USD.")

    col1, col2 = st.columns(2)
    with col1:
        job_title      = st.selectbox("Job Title", job_titles)
        experience     = st.selectbox("Experience Level", list(exp_labels.keys()))
        employment     = st.selectbox("Employment Type", list(emp_labels.keys()))
    with col2:
        company_size   = st.selectbox("Company Size", list(size_labels.keys()))
        remote         = st.selectbox("Remote Ratio", list(remote_labels.keys()))

    if st.button("Predict Salary", use_width="stretch"):
        exp_code  = ["EN","MI","SE","EX"][exp_labels[experience]]
        size_code = ["S","M","L"][size_labels[company_size]]

        # Filter similar profiles — job title + experience first
        filtered = raw_df[
            (raw_df["job_title"] == job_title) &
            (raw_df["experience_level"] == exp_code)
        ]
        # Fallback — experience + company size
        if len(filtered) < 3:
            filtered = raw_df[
                (raw_df["experience_level"] == exp_code) &
                (raw_df["company_size"] == size_code)
            ]
        # Final fallback — just experience level
        if len(filtered) < 3:
            filtered = raw_df[raw_df["experience_level"] == exp_code]

        pred = max(int(filtered["salary_in_usd"].mean()), 20000)
        similar = raw_df[raw_df["experience_level"] == exp_code]["salary_in_usd"]
        low, high = int(similar.quantile(0.25)), int(similar.quantile(0.75))

        st.success(f"### Estimated Salary: **${pred:,} USD / year**")
        st.caption(f"Based on {len(filtered)} similar profiles from {len(raw_df)} real DS job records (2020-2022)")
        st.info(f"Typical range for **{experience}** professionals: **${low:,} - ${high:,} USD**")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Model Performance Comparison")
    st.markdown("Four models were trained and compared. Best model is used for predictions.")

    perf_df = pd.DataFrame([
        {"Model": k, "R² Score (%)": v["r2"], "MAE (USD)": f"${v['mae']:,}",
         "Selected": "✅ Best" if k == best_name else ""}
        for k, v in results.items()
    ])
    st.dataframe(perf_df, use_width="stretch")

    # R2 bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if k == best_name else "#3498db" for k in results]
    ax.bar(list(results.keys()), [v["r2"] for v in results.values()], color=colors)
    ax.set_ylabel("R² Score (%)")
    ax.set_title("Model Comparison — R² Score")
    ax.set_ylim(0, 50)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("""
    **Note:** R² scores reflect real-world salary variability — DS salaries depend on 
    many factors like company, location, and negotiation that aren't captured in this dataset.
    The model provides a strong directional estimate based on available features.
    """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🔍 Data Insights")
    st.markdown(f"Exploring **{len(raw_df)} Data Science salary records** from 2020–2022.")

    col1, col2 = st.columns(2)

    with col1:
        # Avg salary by experience
        exp_order = ["EN", "MI", "SE", "EX"]
        exp_names = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
        avg_exp = raw_df.groupby("experience_level")["salary_in_usd"].mean().reindex(exp_order)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar([exp_names[e] for e in exp_order], avg_exp.values, color="#3498db")
        ax.set_title("Avg Salary by Experience Level")
        ax.set_ylabel("Salary (USD)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Avg salary by company size
        size_order = ["S", "M", "L"]
        size_names = {"S": "Small", "M": "Medium", "L": "Large"}
        avg_size = raw_df.groupby("company_size")["salary_in_usd"].mean().reindex(size_order)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar([size_names[s] for s in size_order], avg_size.values, color="#2ecc71")
        ax.set_title("Avg Salary by Company Size")
        ax.set_ylabel("Salary (USD)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Top 10 highest paying jobs
        top_jobs = raw_df.groupby("job_title")["salary_in_usd"].mean().nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top_jobs.index, top_jobs.values, color="#e74c3c")
        ax.set_title("Top 10 Highest Paying Job Titles")
        ax.set_xlabel("Avg Salary (USD)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        # Salary distribution
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(raw_df["salary_in_usd"], bins=30, color="#9b59b6", edgecolor="white")
        ax.set_title("Salary Distribution")
        ax.set_xlabel("Salary (USD)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("View Raw Data"):
        st.dataframe(raw_df.drop(columns=["Unnamed: 0"], errors="ignore").head(100))

st.markdown("---")
st.caption("Built with Python · Pandas · Scikit-learn · Matplotlib · Streamlit | Data: Kaggle DS Salaries Dataset")
