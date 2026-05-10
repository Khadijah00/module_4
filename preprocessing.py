# =============================================================
# preprocessing.py
# Handles loading, normalizing, and splitting the iris dataset.
# This is the foundation — all other modules depend on this.
# =============================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    """
    Loads the iris dataset, normalizes features to 0-1 range,
    and splits into 80% training and 20% testing sets.

    Returns:
        X_train: training features (120 flowers, 4 features each)
        X_test:  testing features  (30 flowers, 4 features each)
        y_train: training labels   (120 values: 0, 1, or 2)
        y_test:  testing labels    (30 values: 0, 1, or 2)
        class_names: ['setosa', 'versicolor', 'virginica']
    """

    # --- Step 1: Load dataset ---
    iris = load_iris()
    X = iris.data        # shape: (150, 4) — 150 flowers, 4 features
    y = iris.target      # shape: (150,)   — labels 0, 1, 2
    class_names = iris.target_names

    # --- Step 2: Normalize features to 0-1 range ---
    # Each feature divided by its maximum value
    # So all features are on same scale
    # Prevents one feature dominating weight updates
    X_min = X.min(axis=0)   # minimum of each feature column
    X_max = X.max(axis=0)   # maximum of each feature column
    X_normalized = (X - X_min) / (X_max - X_min)

    # --- Step 3: Split 80/20 with shuffle ---
    # random_state=42 means shuffle is same every run
    # so results are reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # --- Step 4: Print summary so we can verify ---
    print("=" * 45)
    print("DATA LOADED AND PREPARED")
    print("=" * 45)
    print(f"Total flowers:        {len(X)}")
    print(f"Training flowers:     {len(X_train)}")
    print(f"Testing flowers:      {len(X_test)}")
    print(f"Features per flower:  {X.shape[1]}")
    print(f"Classes:              {class_names}")
    print()
    print("Feature ranges after normalization:")
    print(f"  Min: {X_normalized.min(axis=0).round(2)}")
    print(f"  Max: {X_normalized.max(axis=0).round(2)}")
    print("=" * 45)
    print()

    return X_train, X_test, y_train, y_test, class_names