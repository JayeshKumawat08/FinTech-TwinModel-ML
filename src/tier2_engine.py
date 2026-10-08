import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "processed", "cleaned_reviews.csv")

    logging.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    df = df.dropna(subset=["cleaned_content"])
    
    # 1. Tier-2 Isolation: Drop compliments, keep only actionable complaints
    df = df[df["score"].isin([1, 2, 3])].copy()
    logging.info(f"Filtered to actionable reviews (scores 1-3): {len(df)} rows")
    logging.info(f"Score distribution:\n{df['score'].value_counts().sort_index()}")

    X = df["cleaned_content"]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 2. Advanced Ensemble Architecture
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])

    logging.info("Training Tier-2 Risk Escalation pipeline...")
    pipeline.fit(X_train, y_train)
    logging.info("Training complete")

    y_pred = pipeline.predict(X_test)

    print("\n=== Tier-2 Classification Report (Scores 1, 2, 3) ===")
    print(classification_report(y_test, y_pred, target_names=["Score 1 (Critical)", "Score 2 (High)", "Score 3 (Medium)"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Tier-2 engine pipeline failed")
        raise