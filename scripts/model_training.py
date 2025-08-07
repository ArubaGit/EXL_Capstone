# scripts/model_training.py

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def get_model(model_name):
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'logistic_regression':
        return LogisticRegression(max_iter=1000, class_weight='balanced')
    elif model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


def train_model(df, model_name='random_forest', target_col='Churn'):
    # 1. Split data
    X = df.drop(columns=[target_col, 'CustomerID'])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_data = df.iloc[y_test.index]  # Keep original test rows for CustomerId

    # 2. Train model
    model = get_model(model_name)
    model.fit(X_train, y_train)

    # 3. Predict
    y_pred = model.predict(X_test)

    # 4. Save model
    os.makedirs('model', exist_ok=True)
    with open(f'model/{model_name}_churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 5. Save metrics
    with open(f'result/test_data/{model_name}_model_metrics.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))


    # 7. Visualizations
    os.makedirs("plots", exist_ok=True)
    pastel_palette = ['#FFB6C1', '#B0E0E6']  # light pink and light blue

    # 7a. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette("seagreen", as_cmap=True),
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()

    # 7b. Churn distribution bar plot
    churn_counts = pd.Series(y_pred).value_counts().sort_index()
    churn_labels = ['No Churn', 'Churn']
    plt.figure()
    sns.barplot(x=churn_labels, y=churn_counts, palette=pastel_palette)
    plt.title(f'{model_name} - Predicted Churn Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_churn_bar.png')
    plt.close()

    return model
