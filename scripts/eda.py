# scripts/eda.py

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set a "cute" color theme
sns.set_style("whitegrid")
sns.set_palette("pastel")

def plot_churn_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Count', fontsize=14)
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.savefig('feature/eda/churn_count.png', bbox_inches='tight')
    plt.clf()

def plot_age_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Age'], kde=True, color=sns.color_palette("muted")[3])
    plt.title('Age Distribution', fontsize=14)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('feature/eda/age_distribution.png', bbox_inches='tight')
    plt.clf()

def plot_correlation_matrix(df):
   
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Handle NaNs and Infs
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_numeric.dropna(inplace=True)

    # Compute correlation
    correlation = df_numeric.corr()

    # Use a visually appealing color map
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig('feature/eda/correlation_matrix.png', bbox_inches='tight')
    plt.clf()
