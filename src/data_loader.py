import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinTechDataLoader:
    """
    Module 1: Data Acquisition Engine. 
    Ingests raw Hinglish banking reviews and enforces strict schema validation.
    """
    
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.text_col = "content"
        self.target_col = "score"

    def load_and_validate(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_data_dir):
            logging.error(f"Directory missing: {self.raw_data_dir}")
            raise FileNotFoundError(f"Directory not found: {self.raw_data_dir}")

        logging.info("Initializing multi-file Pandas ingestion engine...")
        
        all_dataframes = []
        total_files_processed = 0
        
        # Map the exact folder names to their comparative labels
        folders = {
            "Govt_Banks": "Government",
            "Pvt_Banks": "Private"
        }

        for folder_name, bank_type in folders.items():
            folder_path = os.path.join(self.raw_data_dir, folder_name)
            if not os.path.exists(folder_path):
                logging.warning(f"Missing folder: {folder_path}. Please ensure it is placed inside data/raw/")
                continue
            
            # glob searches the folder and grabs every file ending in .csv
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            
            for file in csv_files:
                try:
                    df_temp = pd.read_csv(file)
                    
                    # Guard Clause: Prevent 0-byte ghost files from crashing the pipeline
                    if df_temp.empty:
                        logging.warning(f"{os.path.basename(file)} is completely blank. Skipping.")
                        continue
                        
                    # Inject the Bank_Type label for the Tier-2 comparative analysis
                    df_temp["Bank_Type"] = bank_type
                    all_dataframes.append(df_temp)
                    total_files_processed += 1
                    
                except pd.errors.EmptyDataError:
                    logging.warning(f"{os.path.basename(file)} threw an EmptyDataError. Skipping.")
        
        if not all_dataframes:
            raise ValueError("No valid CSV files found in the specified folders.")

        # Snap all loaded dataframes together into one massive table
        df = pd.concat(all_dataframes, ignore_index=True)
        
        # Schema Validation
        required_cols = {self.text_col, self.target_col, "Bank_Type"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(f"Critical columns missing: {missing_cols}")

        initial_rows = len(df)
        
        # Clean missing text and force string types to eliminate Ghost NaNs
        df = df.dropna(subset=[self.text_col, self.target_col])
        df[self.text_col] = df[self.text_col].astype(str)
        
        dropped_rows = initial_rows - len(df)
        logging.info(f"Ingestion Complete: {len(df)} rows loaded from {total_files_processed} files. ({dropped_rows} empty records dropped).")
        
        return df

# Local execution test
if __name__ == "__main__":
    # Point the path to the main 'raw' directory, NOT a specific file
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    
    loader = FinTechDataLoader(dataset_dir)
    try:
        raw_dataframe = loader.load_and_validate()
        print("\n--- MASTER DATASET SUMMARY ---")
        print(raw_dataframe["Bank_Type"].value_counts())
    except Exception as e:
        print(f"Pipeline Halted: {e}")