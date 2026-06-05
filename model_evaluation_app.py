from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


DATASET_PATH = "AIML Dataset.csv"

MODEL_FILES = {
    "XGBoost": "fraud_detection_xgboost_pipeline.pkl",
    "Logistic Regression": "fraud_detection_logistic_pipeline.pkl",
    "Random Forest": "fraud_detection_random_forest_pipeline.pkl",
}

FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceDiffOrig",
    "balanceDiffDest",
]

NOISE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

SAMPLE_SIZE = 20000
RANDOM_STATE = 42
NOISE_LEVEL = 0.01
ROBUSTNESS_REPEATS = 3


def load_test_data(sample_size, random_state):
    df = pd.read_csv(DATASET_PATH)
    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]

    X = df[FEATURE_COLUMNS]
    y = df["isFraud"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    if sample_size < len(X_test):
        sample_df = X_test.copy()
        sample_df["isFraud"] = y_test.values
        sampled_groups = []
        for _, group in sample_df.groupby("isFraud"):
            group_sample_size = min(
                len(group),
                max(1, int(sample_size * len(group) / len(X_test))),
            )
            sampled_groups.append(group.sample(group_sample_size, random_state=random_state))
        sample_df = pd.concat(sampled_groups)

        remaining = sample_size - len(sample_df)
        if remaining > 0:
            extra = X_test.drop(sample_df.index).sample(remaining, random_state=random_state)
            extra["isFraud"] = y_test.loc[extra.index]
            sample_df = pd.concat([sample_df, extra])

        X_test = sample_df[FEATURE_COLUMNS]
        y_test = sample_df["isFraud"]

    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def load_model(model_path):
    return joblib.load(model_path)


def make_noisy_copy(X, rng, noise_level):
    noisy = X.copy()

    for column in NOISE_COLUMNS:
        scale = np.maximum(noisy[column].abs().to_numpy(), 1.0)
        noise = rng.normal(0, noise_level, size=len(noisy)) * scale
        noisy[column] = np.maximum(noisy[column].to_numpy() + noise, 0)

    noisy["balanceDiffOrig"] = noisy["oldbalanceOrg"] - noisy["newbalanceOrig"]
    noisy["balanceDiffDest"] = noisy["oldbalanceDest"] - noisy["newbalanceDest"]
    return noisy


def robustness_score(model, X, baseline_predictions, random_state, noise_level, repeats):
    rng = np.random.default_rng(random_state)
    consistency_scores = []

    for _ in range(repeats):
        noisy_X = make_noisy_copy(X, rng, noise_level)
        noisy_predictions = model.predict(noisy_X)
        consistency_scores.append(np.mean(noisy_predictions == baseline_predictions))

    return float(np.mean(consistency_scores))


def evaluate_model(model_name, model_path, X_test, y_test, random_state, noise_level, repeats):
    model = load_model(model_path)

    start = perf_counter()
    predictions = model.predict(X_test)
    total_latency = perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    latency_ms = (total_latency / len(X_test)) * 1000
    robustness = robustness_score(
        model,
        X_test,
        predictions,
        random_state=random_state,
        noise_level=noise_level,
        repeats=repeats,
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Latency / Prediction (ms)": latency_ms,
        "Robustness Score": robustness,
    }


def main():
    print("Transaction Fraud Prediction System")
    print("Model Evaluation")
    print("=" * 40)

    missing_models = [
        model_name
        for model_name, model_path in MODEL_FILES.items()
        if not Path(model_path).exists()
    ]

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if missing_models:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing_models))

    print(f"Loading test data from {DATASET_PATH}...")
    X_test, y_test = load_test_data(SAMPLE_SIZE, RANDOM_STATE)
    print(f"Evaluating on {len(X_test):,} stratified test rows.")
    print(f"Robustness noise level: {NOISE_LEVEL:.1%}")
    print(f"Robustness repeats: {ROBUSTNESS_REPEATS}")
    print()

    results = []

    for model_name, model_path in MODEL_FILES.items():
        print(f"Evaluating {model_name}...")
        results.append(
            evaluate_model(
                model_name,
                model_path,
                X_test,
                y_test,
                random_state=RANDOM_STATE,
                noise_level=NOISE_LEVEL,
                repeats=ROBUSTNESS_REPEATS,
            )
        )

    results_df = pd.DataFrame(results)
    ranked_df = results_df.copy()
    ranked_df["Overall Rank Score"] = (
        ranked_df["Accuracy"].rank(ascending=False)
        + ranked_df["F1 Score"].rank(ascending=False)
        + ranked_df["Latency / Prediction (ms)"].rank(ascending=True)
        + ranked_df["Robustness Score"].rank(ascending=False)
    )
    best_model = ranked_df.sort_values("Overall Rank Score").iloc[0]["Model"]

    display_df = results_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].map("{:.4f}".format)
    display_df["F1 Score"] = display_df["F1 Score"].map("{:.4f}".format)
    display_df["Latency / Prediction (ms)"] = display_df["Latency / Prediction (ms)"].map("{:.4f}".format)
    display_df["Robustness Score"] = display_df["Robustness Score"].map("{:.4f}".format)

    print()
    print("Evaluation Results")
    print("-" * 40)
    print(display_df.to_string(index=False))
    print()
    print(f"Best overall model by combined ranking: {best_model}")


if __name__ == "__main__":
    main()
