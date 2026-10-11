import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.text_pipeline import HinglishTextCleaner

st.set_page_config(page_title="FinTech Twin-Model Dashboard", layout="wide", initial_sidebar_state="expanded")

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_resource
def load_models():
    tier1 = joblib.load(MODELS_DIR / "tier1_pipeline.pkl")
    tier2 = joblib.load(MODELS_DIR / "tier2_pipeline.pkl")
    cleaner = HinglishTextCleaner()
    return tier1, tier2, cleaner


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "cleaned_reviews.csv")
    df = df.dropna(subset=["cleaned_content"])
    df["is_actionable"] = df["score"].apply(lambda x: 1 if x <= 3 else 0)
    return df


tier1_pipeline, tier2_pipeline, text_cleaner = load_models()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Risk Engine", "Dataset Analytics", "Model Metrics"])


if page == "Live Risk Engine":
    st.title("Live Risk Engine")
    st.markdown("Enter a raw Hinglish banking review to see the Twin-Model cascade in action.")

    user_input = st.text_area(
        "Review Text (Hinglish/English)",
        placeholder="e.g., 'UPI payment failed 3 times, customer care useless, paisa kat gaya but transaction pending'",
        height=120,
    )

    if st.button("Analyze Risk", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a review text.")
        else:
            with st.spinner("Processing through pipeline..."):
                cleaned = text_cleaner.transform(pd.Series([user_input])).iloc[0]

                tier1_pred = tier1_pipeline.predict([cleaned])[0]
                tier1_proba = tier1_pipeline.predict_proba([cleaned])[0]
                actionable_conf = tier1_proba[1] if tier1_pred == 1 else tier1_proba[0]

                st.subheader("Step 1: Text Cleaning")
                st.code(cleaned, language=None)

                st.subheader("Step 2: Tier-1 Actionability Filter")
                if tier1_pred == 1:
                    st.success(f"Actionable Complaint (Confidence: {actionable_conf:.1%})")
                else:
                    st.info(f"Non-Actionable / Spam (Confidence: {actionable_conf:.1%})")

                st.subheader("Step 3: Tier-2 Risk Escalation")
                if tier1_pred == 1:
                    tier2_pred = tier2_pipeline.predict([cleaned])[0]
                    tier2_proba = tier2_pipeline.predict_proba([cleaned])[0]
                    conf = tier2_proba[tier2_pred - 1]

                    severity_map = {1: ("Critical", "error"), 2: ("High", "warning"), 3: ("Medium", "info")}
                    label, banner_type = severity_map[tier2_pred]
                    getattr(st, banner_type)(f"Severity Score: {tier2_pred} — {label} Risk (Confidence: {conf:.1%})")

                    prob_df = pd.DataFrame({
                        "Severity": ["Score 1 (Critical)", "Score 2 (High)", "Score 3 (Medium)"],
                        "Probability": tier2_proba
                    })
                    fig = px.bar(prob_df, x="Severity", y="Probability", title="Tier-2 Class Probabilities",
                                 color="Severity", color_discrete_map={
                                     "Score 1 (Critical)": "#ff4b4b",
                                     "Score 2 (High)": "#ffa500",
                                     "Score 3 (Medium)": "#1f77b4"
                                 })
                    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tier-2 skipped: Review classified as Non-Actionable.")


elif page == "Dataset Analytics":
    st.title("Dataset Analytics")
    st.markdown("Exploratory analysis of the cleaned FinTech review corpus.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Actionable (1-3★)", f"{(df['is_actionable'] == 1).sum():,}")
    col3.metric("Banks Tracked", df["Bank_Type"].nunique())

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Review Severity by Bank Type")
        fig_hist = px.histogram(
            df, x="score", color="Bank_Type", barmode="stack",
            category_orders={"score": [1, 2, 3, 4, 5]},
            labels={"score": "Star Rating", "count": "Number of Reviews"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_hist.update_layout(xaxis_title="Star Rating", yaxis_title="Count", legend_title="Bank Type")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("Actionability → Severity Hierarchy")
        df_sunburst = df.copy()
        df_sunburst["actionable_label"] = df_sunburst["is_actionable"].map({1: "Actionable (1-3★)", 0: "Non-Actionable (4-5★)"})
        fig_sun = px.sunburst(
            df_sunburst, path=["actionable_label", "score"],
            color="score", color_continuous_scale="RdYlGn_r",
            labels={"score": "Star Rating"}
        )
        fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)


elif page == "Model Metrics":
    st.title("Model Metrics & Experiments")
    st.markdown("Performance benchmarks for the Twin-Model architecture.")

    st.subheader("Tier-1: Actionability Classifier")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=94,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Test Accuracy"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 80], 'color': "#fff3e0"},
                {'range': [80, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 94}
        }
    ))
    gauge.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(gauge, use_container_width=True)

    st.caption("Tier-1 Logistic Regression (TF-IDF 5000) — 80/20 stratified split on 639K reviews.")

    st.divider()

    st.subheader("Tier-2: Severity Classification Experiments")
    st.markdown("Comparing Standard Random Forest vs. LSA-Compressed (TruncatedSVD) Random Forest on actionable complaints (Scores 1-3).")

    metrics_df = pd.DataFrame({
        "Model": ["Standard RF", "Standard RF", "Standard RF", "RF + TruncatedSVD", "RF + TruncatedSVD", "RF + TruncatedSVD"],
        "Severity": ["Score 1 (Critical)", "Score 2 (High)", "Score 3 (Medium)"] * 2,
        "F1_Score": [0.89, 0.10, 0.72, 0.87, 0.08, 0.68]
    })

    fig_bar = px.bar(
        metrics_df, x="Severity", y="F1_Score", color="Model", barmode="group",
        text_auto=".0%", color_discrete_map={
            "Standard RF": "#1f77b4",
            "RF + TruncatedSVD": "#ff7f0e"
        }
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", yaxis_title="F1-Score", xaxis_title="Severity Class")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.warning(
        "**Key Finding:** The 'Muddy Middle' (Score 2) remains the hardest class. "
        "LSA compression (3000 → 300 dims) slightly degraded all classes, with Score 2 dropping from 10% to 8% F1. "
        "This suggests the Score 2 confusion is semantic (genuine ambiguity) rather than representational noise."
    )

    with st.expander("Experiment Details"):
        st.markdown("""
        - **Standard RF**: TF-IDF(3000, ngram=1-2) → RandomForest(100, balanced)
        - **RF + TruncatedSVD**: TF-IDF(3000, ngram=1-2) → TruncatedSVD(300) → RandomForest(100, balanced)
        - Train/Test: 80/20 stratified on scores 1,2,3 only (~400K rows)
        - Metric: Macro F1 per class
        """)

st.sidebar.divider()
st.sidebar.caption("FinTech Twin-Model v1.0 | Enterprise ML Dashboard")