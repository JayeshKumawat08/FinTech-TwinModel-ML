import os
import logging
import pandas as pd

from data_loader import FinTechDataLoader
from text_pipeline import HinglishTextCleaner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    # 1. Dynamically locate the project root folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    output_path = os.path.join(processed_dir, "cleaned_reviews.csv")

    # 2. Ensure the vault exists
    os.makedirs(processed_dir, exist_ok=True)
    logging.info(f"Output directory ensured: {processed_dir}")

    # 3. Ignite Phase 1: Load the 639,415 rows
    loader = FinTechDataLoader(raw_dir)
    df = loader.load_and_validate()
    logging.info(f"Raw data loaded: {len(df)} rows")

    # 4. Ignite Phase 2: Scrub the text
    cleaner = HinglishTextCleaner()
    df["cleaned_content"] = cleaner.fit_transform(df["content"])
    logging.info("Text cleaning complete")

    # 5. Drop the old uncleaned text column to save RAM
    df = df[["cleaned_content", "score", "Bank_Type"]]
    logging.info(f"Columns retained: {list(df.columns)}")

    # 6. Save the final pristine matrix
    df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to: {output_path}")
    logging.info(f"Final shape: {df.shape}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Preprocessing pipeline failed")
        raise