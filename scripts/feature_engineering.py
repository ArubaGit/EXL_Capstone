# scripts/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=['Gender'], drop_first=True)

def scale_features(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
