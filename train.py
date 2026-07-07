"""
MLflow Experiment Tracking Assignment
--------------------------------------
Trains 3 classifiers (Logistic Regression, Decision Tree, Random Forest) on the
Iris dataset, each with different hyperparameters, and tracks every run with
MLflow: params, metrics (accuracy, precision, recall, f1), confusion-matrix
artifact, and the saved model itself.

Run with:
    python train.py

Then view results with:
    mlflow ui
    (open http://127.0.0.1:5000 in your browser)
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

EXPERIMENT_NAME = "iris-classification"


# ---------------------------------------------------------------------------
# Task 2: Data preparation
# ---------------------------------------------------------------------------
def load_data(test_size=0.2, random_state=42):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    X = df[iris.feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, iris.target_names


# ---------------------------------------------------------------------------
# Task 4: Train + log a single run
# ---------------------------------------------------------------------------
def run_experiment(model_name, model, params, X_train, X_test, y_train, y_test,
                    target_names, run_index=None):
    run_label = f"{model_name}" if run_index is None else f"{model_name}_run{run_index}"

    with mlflow.start_run(run_name=run_label):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Log params (model name + hyperparameters)
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {run_label}")
        cm_path = f"confusion_matrix_{run_label}.png"
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        # Save the trained model
        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")

        print(f"[{run_label}] accuracy={acc:.4f} precision={prec:.4f} "
              f"recall={rec:.4f} f1={f1:.4f}")

        return {
            "run": run_label,
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, target_names = load_data()

    # Task 3: Three models, three different hyperparameter sets
    experiments = [
        (
            "LogisticRegression",
            LogisticRegression(C=1.0, max_iter=200, solver="lbfgs"),
            {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42),
            {"max_depth": 4, "criterion": "gini", "random_state": 42},
        ),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        ),
    ]

    results = []
    for model_name, model, params in experiments:
        result = run_experiment(
            model_name, model, params, X_train, X_test, y_train, y_test, target_names
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    print("\n=== Summary of Core Runs ===")
    print(results_df.to_string(index=False))

    best = results_df.loc[results_df["f1_score"].idxmax()]
    print(f"\nBest model: {best['model']} (f1_score={best['f1_score']:.4f})")


if __name__ == "__main__":
    main()
