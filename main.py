# main.py

from database.mysqldb import connect_db, display_all_customers, insert_data_from_csv
from scripts.data_loader import load_csv_data
from scripts.data_cleaner import clean_data
from scripts.eda import plot_churn_distribution, plot_age_distribution, plot_correlation_matrix
from scripts.feature_engineering import one_hot_encode, scale_features
from scripts.model_training import train_model

import os

if __name__ == "__main__":

    file_path = "data/raw/exl_credit_card_churn_data.csv"
    #SQL
    connect_db()
    # insert_data_from_csv(file_path)
    display_all_customers()

    # 1. Load raw data
    df = load_csv_data(file_path)

    if df.empty:
        print("Failed to load data. Exiting.")
        exit()
    print("_________________DataFrame shape:_________________\n", df.shape)
    print("_________________Data types:_____________________\n", df.dtypes)
    print("_________________Head of data:___________________\n", df.head())
    # 2. Clean data
    df_clean = clean_data(df)

    # Save cleaned data to processed folder
    df_clean.to_csv("data/processed/churn_cleaned.csv", index=False)
    print("Cleaned data saved to data/processed/churn_cleaned.csv")
    print("___________________________CLEANED DATA____________________________________")
    print(df_clean.head())

    # 3. Run EDA and save plots
    os.makedirs("feature/eda", exist_ok=True)
    plot_churn_distribution(df_clean)
    plot_age_distribution(df_clean)
    plot_correlation_matrix(df_clean)

    # 4. Encode and scale
    df_encoded = one_hot_encode(df_clean)
    df_encoded.to_csv("data/processed/churn_encoded.csv", index=False)
    print("_____________________________DATA AFTER ONE HOT ENCODING_________________________")
    print(df_encoded.head())

    numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df_scaled = scale_features(df_encoded, numeric_cols)
    df_scaled.to_csv("data/processed/churn_scaled.csv", index=False)
    print("_____________________________DATA AFTER SCALING________________________________")
    print(df_scaled.head())

    # Check class distribution
    print("______________Class distribution after scaling:_________________________")
    print(df_scaled['Churn'].value_counts(normalize=True))


    models_to_train = ['knn', 'random_forest', 'logistic_regression']

    for model_name in models_to_train:
        print(f"\n_________________ Training {model_name.upper()} model _____________________")
        model = train_model(df_scaled, model_name)

    print(" Pipeline complete.")
