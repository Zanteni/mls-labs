"""
Evaluates the student's trained model (outputs/theta.npy, saved by the
notebook itself) against the hidden test set.

Usage:
    python grading/grade_performance.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
THETA_PATH = os.path.join(REPO_ROOT, "outputs", "theta.npy")
HIDDEN_TEST_PATH = os.path.join(REPO_ROOT, "instructor_private", "test_hidden.csv")


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def score_from_r2(r2):
    """Map R2 to a 0-100 performance score band."""
    if r2 > 0.90:
        return 100
    elif r2 > 0.80:
        return 85
    elif r2 > 0.60:
        return 60
    elif r2 > 0.30:
        return 30
    else:
        return 0


def main():
    if not os.path.exists(THETA_PATH):
        print(f"ERROR: {THETA_PATH} not found. Did the notebook run and save theta?")
        sys.exit(1)

    theta = np.load(THETA_PATH)

    df = pd.read_csv(HIDDEN_TEST_PATH)
    X_raw = df[["x1", "x2"]].values
    y_true = df["y"].values
    X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

    if theta.shape[0] != X.shape[1]:
        print(f"ERROR: theta shape {theta.shape} does not match expected {X.shape[1]}")
        sys.exit(1)

    y_pred = X @ theta
    r2 = r2_score(y_true, y_pred)
    mse_val = mse(y_true, y_pred)
    perf_score = score_from_r2(r2)

    result = {
        "r2": round(float(r2), 4),
        "mse": round(float(mse_val), 4),
        "performance_score": perf_score,
    }

    with open(os.path.join(REPO_ROOT, "outputs", "performance_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
