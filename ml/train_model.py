"""
WeatherGPT - Rain Prediction Model Training Script
Trains a RandomForestClassifier on Iyerpadi weather dataset to predict rain.
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
import joblib

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "01_Iyerpadi_Weather_Data.csv"
MODEL_PATH = BASE_DIR / "weather_model.pkl"

# Feature and target columns
FEATURE_COLUMNS = [
    "temp",
    "RH",
    "windSpeed",
    "gustSpeed",
    "DewPt",
    "windDirection",
    "PAR",
]
TARGET_COLUMN = "rain"


def load_and_preprocess_data(dataset_path: Path):
    """
    Loads weather dataset, removes missing values, and prepares features & binary target.
    """
    print(f"Loading dataset from: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    total_raw = len(df)

    # Remove missing values across required features and target
    clean_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    total_used = len(clean_df)
    dropped_count = total_raw - total_used

    print(f"Total raw records in CSV: {total_raw:,}")
    print(f"Missing records dropped:  {dropped_count:,}")
    print(f"Number of records used:   {total_used:,}")

    # Features (X)
    X = clean_df[FEATURE_COLUMNS]

    # Target (y): Binary classification
    # rain > 0 = 1 (Rain), rain == 0 = 0 (No Rain)
    y = (clean_df[TARGET_COLUMN] > 0).astype(int)

    no_rain_count = int((y == 0).sum())
    rain_count = int((y == 1).sum())
    print(
        f"Target distribution: No Rain (0) = {no_rain_count:,} ({no_rain_count / total_used * 100:.2f}%), "
        f"Rain (1) = {rain_count:,} ({rain_count / total_used * 100:.2f}%)"
    )

    return X, y, total_used


def train_and_evaluate():
    """
    Trains the RandomForest model, evaluates metrics, and saves the trained model.
    """
    print("=" * 60)
    print("WeatherGPT: Training Rain Prediction Model")
    print("=" * 60)

    # 1. Load and clean data
    X, y, total_records_used = load_and_preprocess_data(DATASET_PATH)

    # 2. Split dataset into 80% training and 20% testing
    print("\nSplitting dataset into 80% train and 20% test (stratify=y, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records:  {len(X_test):,}")

    # 3. Initialize and train RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    print("Parameters:")
    print("  - n_estimators: 100")
    print("  - random_state: 42")
    print("  - n_jobs: -1")
    print("  - class_weight: 'balanced'")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 4. Model Evaluation
    print("\nEvaluating model on testing set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=["No Rain (0)", "Rain (1)"],
        digits=4,
    )

    # 5. Display required evaluation statistics
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of records used: {total_records_used:,}")
    print(f"Training records:       {len(X_train):,}")
    print(f"Testing records:        {len(X_test):,}")
    print(f"Accuracy:               {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision:              {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:                 {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:               {f1:.4f} ({f1 * 100:.2f}%)")
    print("\nClassification Report:")
    print(clf_report)
    print("=" * 60)

    # 6. Save model to ml/weather_model.pkl
    print(f"\nSaving trained model to: {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_and_evaluate()
