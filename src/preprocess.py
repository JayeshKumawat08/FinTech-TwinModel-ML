import os
import logging
import pandas as pd

from src.data_loader import FinTechDataLoader
from src.text_pipeline import HinglishTextCleaner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    output_path = os.path.join(processed_dir, "cleaned_reviews.csv")

    os.makedirs(processed_dir, exist_ok=True)
    logging.info(f"Output directory ensured: {processed_dir}")

    loader = FinTechDataLoader(raw_dir)
    df = loader.load_and_validate()
    logging.info(f"Raw data loaded: {len(df)} rows")

    cleaner = HinglishTextCleaner()
    df["cleaned_content"] = cleaner.fit_transform(df["content"])
    logging.info("Text cleaning complete")

    df = df[["cleaned_content", "score", "Bank_Type"]]
    logging.info(f"Columns retained: {list(df.columns)}")

    df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to: {output_path}")
    logging.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Preprocessing pipeline failed")
        raise