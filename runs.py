"""
Bonus Challenge: 10+ MLflow runs, varying model type, hyperparameters,
train-test split, and random state. At the end, builds a comparison table
(Run, Model, Accuracy, Precision, Recall, F1-score) and saves it as CSV +
prints it, and also saves it as an MLflow artifact on a dedicated summary run.

Run with:
    python bonus_runs.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

EXPERIMENT_NAME = "iris-classification"

iris = load_iris()
FEATURE_NAMES = iris.feature_names
TARGET_NAMES = iris.target_names
X_full = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
y_full = pd.Series(iris.target)

# 12 configurations: varying model, hyperparameters, test_size, random_state
CONFIGS = [
    ("LogisticRegression", LogisticRegression, {"C": 0.1, "max_iter": 200}, 0.2, 0),
    ("LogisticRegression", LogisticRegression, {"C": 10.0, "max_iter": 300}, 0.3, 1),
    ("DecisionTree", DecisionTreeClassifier, {"max_depth": 2, "criterion": "gini"}, 0.2, 0),
    ("DecisionTree", DecisionTreeClassifier, {"max_depth": 6, "criterion": "entropy"}, 0.25, 7),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 50, "max_depth": 3}, 0.2, 0),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 200, "max_depth": 8}, 0.3, 5),
    ("KNN", KNeighborsClassifier, {"n_neighbors": 3}, 0.2, 0),
    ("KNN", KNeighborsClassifier, {"n_neighbors": 7, "weights": "distance"}, 0.25, 3),
    ("SVM", SVC, {"C": 1.0, "kernel": "linear"}, 0.2, 0),
    ("SVM", SVC, {"C": 5.0, "kernel": "rbf"}, 0.3, 2),
    ("NaiveBayes", GaussianNB, {}, 0.2, 0),
    ("NaiveBayes", GaussianNB, {"var_smoothing": 1e-8}, 0.35, 9),
]


def build_model(cls, params):
    # random_state isn't a valid param for GaussianNB / default KNN configs
    if "random_state" in cls().get_params() and "random_state" not in params:
        params = {**params, "random_state": 42}
    return cls(**params)


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    summary_rows = []

    for i, (model_name, cls, params, test_size, split_seed) in enumerate(CONFIGS, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=split_seed, stratify=y_full
        )

        model = build_model(cls, params)
        run_label = f"run{i:02d}_{model_name}"

        with mlflow.start_run(run_name=run_label):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("split_random_state", split_seed)
            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"Confusion Matrix - {run_label}")
            cm_path = f"confusion_matrix_{run_label}.png"
            fig.savefig(cm_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(cm_path)

            mlflow.sklearn.log_model(
                model, artifact_path="model", serialization_format="pickle"
            )

            print(f"[{run_label}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

            summary_rows.append({
                "Run": run_label,
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-score": round(f1, 4),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("comparison_table.csv", index=False)

    print("\n=== Comparison Table (all runs) ===")
    print(summary_df.to_string(index=False))

    best_row = summary_df.loc[summary_df["F1-score"].idxmax()]
    worst_row = summary_df.loc[summary_df["F1-score"].idxmin()]
    print(f"\nBest run:  {best_row['Run']} ({best_row['Model']}) - F1={best_row['F1-score']}")
    print(f"Worst run: {worst_row['Run']} ({worst_row['Model']}) - F1={worst_row['F1-score']}")

    # Log the comparison table itself as an artifact on one more tracked run
    with mlflow.start_run(run_name="bonus_comparison_summary"):
        mlflow.log_artifact("comparison_table.csv")
        mlflow.log_metric("num_runs_compared", len(summary_df))
        mlflow.log_metric("best_f1", best_row["F1-score"])
        mlflow.log_metric("worst_f1", worst_row["F1-score"])


if __name__ == "__main__":
    main()
