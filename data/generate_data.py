"""
Run this ONCE (as the instructor) to generate the datasets for the lab.

- train.csv        -> goes into the student template repo
- test_hidden.csv  -> KEEP THIS PRIVATE. Used only by grade_performance.py
                      to score students on unseen data.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_TRAIN = 200
N_HIDDEN = 60
N_FEATURES = 2

# Ground-truth relationship: y = 4 + 3*x1 - 2*x2 + noise
true_theta = np.array([4.0, 3.0, -2.0])  # [bias, w1, w2]


def make_dataset(n):
    X_raw = np.random.uniform(-5, 5, size=(n, N_FEATURES))
    X_bias = np.hstack([np.ones((n, 1)), X_raw])  # add bias column
    noise = np.random.normal(0, 1.0, size=n)
    y = X_bias @ true_theta + noise
    return X_raw, y


def save(path, X_raw, y):
    df = pd.DataFrame(X_raw, columns=[f"x{i+1}" for i in range(N_FEATURES)])
    df["y"] = y
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")


if __name__ == "__main__":
    X_train, y_train = make_dataset(N_TRAIN)
    X_hidden, y_hidden = make_dataset(N_HIDDEN)

    save("train.csv", X_train, y_train)
    save("test_hidden.csv", X_hidden, y_hidden)

    print("\nTrue theta (for your reference only):", true_theta)
    print("Remember: test_hidden.csv must NEVER be committed to the student repo.")
