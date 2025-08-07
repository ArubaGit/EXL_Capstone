import pandas as pd
from scripts.data_loader import load_csv_data
from scripts.eda import plot_churn_distribution, plot_age_distribution, plot_correlation_matrix
from scripts.feature_engineering import one_hot_encode, scale_features
from scripts.model_training import train_model


test_data = pd.DataFrame({
    'CustomerId': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009],
    'Gender': ['Male', 'Female', 'Female', 'male', 'FEMALE', 'female', 'male', 'female', 'male'],
    'Age': [35, 42, 28, -5, 51, 30, 45, 38, 60],  # -5 is invalid
    'Tenure': [3, 5, 2, 1, 7, 4, 6, 3, 8],
    'Balance': [50000.0, 0.0, -100.0, 30000.0, 75000.0, 25000.0, 60000.0, 40000.0, 85000.0],  # -100 is invalid
    'NumOfProducts': [2, 1, 1, 3, 2, 2, 1, 2, 2],
    'HasCrCard': [1, 0, 1, 1, 0, 1, 1, 0, 1],
    'IsActiveMember': [1, 0, 1, 1, 0, 1, 1, 0, 1],
    'EstimatedSalary': [60000, 52000, 45000, 72000, 68000, 56000, 47000, 63000, 75000],
    'Churn': ['1', '0', 'yes', 'No', '1', '0', '1', '0', '1']  # 'yes'/'No' are inconsistent
})

def clean_data(df):
    df = df.copy()

    # Fix case for Gender and Churn
    df['Gender'] = df['Gender'].str.lower().str.strip()
    df['Churn'] = df['Churn'].astype(str).str.lower().str.strip()

    # Clean invalid ages and balances
    df = df[df['Age'] > 0]
    df = df[df['Balance'] >= 0]

    # Normalize churn labels
    churn_map = {'yes': 1, '1': 1, 'no': 0, '0': 0}
    df['Churn'] = df['Churn'].map(churn_map)

    # Keep only known genders
    df = df[df['Gender'].isin(['male', 'female'])]

    # Drop rows with missing target
    df = df.dropna(subset=['Churn'])

    # Convert to int
    df['Churn'] = df['Churn'].astype(int)

    return df

df_test_clean = clean_data(test_data)
print(df_test_clean.head())
df_test_encoded = one_hot_encode(df_test_clean)
print(df_test_encoded.head())
numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df_test_scaled = scale_features(df_test_encoded, numeric_cols)
print(df_test_scaled.head())


import joblib

# Drop unneeded columns
X_test = df_test_scaled.drop(columns=['CustomerId', 'Churn'])
print ("Test features: ")
print(X_test.head())
y_test = df_test_scaled['Churn']

# Load your trained model
model = joblib.load('model/random_forest_churn_model.pkl')

# Predict
predictions = model.predict(X_test)

# Output
for i, pred in enumerate(predictions):
    print(f"Customer ID: {test_data.iloc[i]['CustomerId']} --> Predicted Churn: {pred} | Actual: {y_test.iloc[i]}")

import os

# Prepare output directory
os.makedirs("result", exist_ok=True)

# Define output file name
model_name = model.__class__.__name__  # e.g., KNeighborsClassifier
output_file = f"result/test_sample/{model_name.lower()}_test_result.txt"

# Write predictions to the file
with open(output_file, "w") as f:
    for i, pred in enumerate(predictions):
        line = f"Customer ID: {df_test_clean.iloc[i]['CustomerId']} --> Predicted Churn: {pred} | Actual: {y_test.iloc[i]}"
        f.write(line + "\n")