import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "processed", "cleaned_reviews.csv")

    logging.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    initial_rows = len(df)
    # Drop rows where cleaning left nothing but empty space
    df = df.dropna(subset=["cleaned_content"])
    dropped = initial_rows - len(df)
    if dropped:
        logging.warning(f"Dropped {dropped} rows with NaN cleaned_content")

    # 1. Feature Engineering: Define what needs bank intervention
    df["is_actionable"] = df["score"].apply(lambda x: 1 if x <= 3 else 0)
    logging.info(f"Target distribution:\n{df['is_actionable'].value_counts()}")

    X = df["cleaned_content"]
    y = df["is_actionable"]

    # 2. Mathematical Split: 80% Training, 20% Testing Exam
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Model Architecture: Bridge Vectorization and Classification
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )),
    ])

    logging.info("Training Tier-1 filter pipeline...")
    pipeline.fit(X_train, y_train)
    logging.info("Training complete")

    # 4. Evaluation: Administer the final exam to the model
    y_pred = pipeline.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Non-Actionable (4-5)", "Actionable (1-3)"]))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Tier-1 filter pipeline failed")
        raise