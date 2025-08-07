import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Show initial row count
    print(f"Original rows: {df.shape[0]}")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Keep only rows with valid binary values
    df = df[df['Churn'].isin(['0', '1'])]
    df = df[df['HasCrCard'].isin(['0', '1'])]
    df = df[df['IsActiveMember'].isin(['0', '1'])]

    # Normalize gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.lower()

    # Convert numerical columns
    numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop missing values
    df = df.dropna()

    # Remove invalid values
    df = df[df['Age'] >= 0]
    df = df[df['Tenure'] >= 0]
    df = df[df['Balance'] >= 0]
    df = df[df['NumOfProducts'] > 0]
    df = df[df['EstimatedSalary'] >= 0]

    # Outlier removal using IQR method
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            original_count = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed = original_count - df.shape[0]
            if removed > 0:
                print(f"Removed {removed} outliers from '{col}'")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Final row count
    print(f"Rows after cleaning: {df.shape[0]}")
    
    return df
