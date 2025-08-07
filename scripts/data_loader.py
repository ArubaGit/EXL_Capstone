# scripts/data_loader.py

import pandas as pd

# Load credit card churn data from a CSV file into a Pandas DataFrame.
def load_csv_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()
