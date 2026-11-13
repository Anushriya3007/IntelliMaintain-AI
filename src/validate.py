
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load training dataset
train_path = BASE_DIR / "data" / "processed" / "train.csv"
df = pd.read_csv(train_path)

# Separate features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Load trained model pipelines
rf_model = joblib.load(BASE_DIR / "models" / "random_forest.pkl")
xgb_model = joblib.load(BASE_DIR / "models" / "xgboost_model.pkl")

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluation metrics
scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

results = []

print("\nINTELLIMAINTAIN-AI MODEL VALIDATION")
print("=" * 50)

for name, model in models.items():

    print(f"\nValidating {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]

        row[f"{metric.upper()} Mean"] = values.mean()
        row[f"{metric.upper()} Std"] = values.std()

        print(
            f"{metric.upper():12}: "
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )

    results.append(row)

# Save validation results
results_df = pd.DataFrame(results)

output_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_validation_cv.csv"
)

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 50)
print("CROSS-VALIDATION COMPLETED!")
print(f"Results saved to: {output_path}")
print("=" * 50)
