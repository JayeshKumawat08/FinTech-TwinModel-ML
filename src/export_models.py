import os
import logging
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "processed", "cleaned_reviews.csv")
    models_dir = os.path.join(project_root, "models")

    os.makedirs(models_dir, exist_ok=True)
    logging.info(f"Models directory ensured: {models_dir}")

    logging.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    df = df.dropna(subset=["cleaned_content"])
    logging.info(f"Data loaded: {len(df)} rows after dropping NaNs")

    # --- Tier-1: Actionable vs Non-Actionable ---
    logging.info("Training Tier-1 pipeline (full dataset)...")
    df_tier1 = df.copy()
    df_tier1["is_actionable"] = df_tier1["score"].apply(lambda x: 1 if x <= 3 else 0)

    tier1_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )),
    ])

    tier1_pipeline.fit(df_tier1["cleaned_content"], df_tier1["is_actionable"])

    tier1_path = os.path.join(models_dir, "tier1_pipeline.pkl")
    joblib.dump(tier1_pipeline, tier1_path)
    logging.info(f"Tier-1 pipeline saved to: {tier1_path}")

    # --- Tier-2: Risk Escalation (Scores 1, 2, 3) ---
    logging.info("Training Tier-2 pipeline (filtered dataset)... This will take a moment.")
    df_tier2 = df[df["score"].isin([1, 2, 3])].copy()
    logging.info(f"Tier-2 training rows: {len(df_tier2)}")
    logging.info(f"Score distribution:\n{df_tier2['score'].value_counts().sort_index()}")

    tier2_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])

    tier2_pipeline.fit(df_tier2["cleaned_content"], df_tier2["score"])

    tier2_path = os.path.join(models_dir, "tier2_pipeline.pkl")
    joblib.dump(tier2_pipeline, tier2_path)
    logging.info(f"Tier-2 pipeline saved to: {tier2_path}")

    logging.info("Both pipelines exported successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Model export failed")
        raise