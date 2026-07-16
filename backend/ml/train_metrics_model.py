"""
Offline training script for the profile-metrics classifier.

Not deployed anywhere. Run this locally whenever the dataset changes:

    python backend/ml/train_metrics_model.py

It trains a logistic regression on a public, labeled Instagram
fake/spammer/genuine-accounts dataset and writes the learned coefficients,
intercept, and feature normalization stats to
frontend/lib/metricsModel.json. frontend/lib/metricsModel.ts re-implements
the same math (normalize -> dot product -> sigmoid) in TypeScript so the
Next.js app can score a profile without calling back into Python.

Dataset: "Instagram fake spammer genuine accounts"
(Kaggle: free4ever1/instagram-fake-spammer-genuine-accounts), mirrored here
as backend/data/instagram_fake_accounts/{train,test}.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "instagram_fake_accounts"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "frontend" / "lib" / "metricsModel.json"

# Dataset columns, in the order the CSVs use them.
FEATURE_COLUMNS = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
]
LABEL_COLUMN = "fake"

# Names used in the exported JSON / TS port - map 1:1 with FEATURE_COLUMNS,
# but as identifiers a TS object can use as keys.
FEATURE_KEYS = [
    "hasProfilePic",
    "usernameDigitRatio",
    "fullnameWordCount",
    "fullnameDigitRatio",
    "nameEqualsUsername",
    "descriptionLength",
    "hasExternalUrl",
    "isPrivate",
    "postCount",
    "followerCount",
    "followCount",
]


def load_split(filename: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_DIR / filename)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy(dtype=float)
    return X, y


def main() -> None:
    X_train, y_train = load_split("train.csv")
    X_test, y_test = load_split("test.csv")

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0  # guard against a constant column

    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_norm, y_train)

    y_pred = model.predict(X_test_norm)
    y_proba = model.predict_proba(X_test_norm)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
    }
    print("Real (not synthetic) held-out test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Note: model predicts P(fake). The app wants a "trust" probability,
    # so metricsModel.ts will report (1 - P(fake)) as the trust score.
    output = {
        "featureKeys": FEATURE_KEYS,
        "featureColumns": FEATURE_COLUMNS,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "metrics": metrics,
        "trainedOn": "Kaggle free4ever1/instagram-fake-spammer-genuine-accounts",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote weights to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
