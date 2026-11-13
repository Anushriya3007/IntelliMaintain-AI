
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cross-validation results
cv_path = BASE_DIR / "data" / "processed" / "model_validation_cv.csv"
cv_df = pd.read_csv(cv_path)

# Load test dataset
test_path = BASE_DIR / "data" / "processed" / "test.csv"
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["Machine failure"])
y_test = test_df["Machine failure"]

# Load models
rf_model = joblib.load(BASE_DIR / "models" / "random_forest.pkl")
xgb_model = joblib.load(BASE_DIR / "models" / "xgboost_model.pkl")

# Evaluate test predictions
models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

test_results = {}

for name, model in models.items():
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    test_results[name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "Brier Score": brier_score_loss(y_test, probabilities),
        "Confusion Matrix": confusion_matrix(y_test, predictions).tolist()
    }

# Create report
report = []

report.append("# IntelliMaintain-AI: Model Evaluation Report")
report.append("")
report.append("## 1. Project Overview")
report.append(
    "IntelliMaintain-AI is a machine learning-based predictive "
    "maintenance system designed to identify potential machine failures "
    "using operational sensor measurements."
)
report.append("")
report.append("## 2. Dataset")
report.append("- Dataset: UCI AI4I 2020 Predictive Maintenance Dataset")
report.append(f"- Test samples: {len(y_test)}")
report.append(f"- Training samples: {len(pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train.csv'))}")
report.append(f"- Failure cases in test set: {int(y_test.sum())}")
report.append(f"- Normal cases in test set: {int((y_test == 0).sum())}")
report.append("")
report.append("## 3. Five-Fold Cross-Validation")
report.append(cv_df.to_markdown(index=False))
report.append("")
report.append("## 4. Test-Set Performance")

for name, metrics in test_results.items():
    report.append(f"### {name}")
    for metric, value in metrics.items():
        report.append(f"- **{metric}:** {value}")
    report.append("")

report.append("## 5. Model Selection")
report.append(
    "XGBoost is the current preferred model because it achieved stronger "
    "cross-validation precision, F1-score and PR-AUC than Random Forest. "
    "Final deployment decisions should also consider false negatives, "
    "false positives and operational requirements."
)
report.append("")
report.append("## 6. Probability Calibration")
report.append(
    f"- XGBoost test-set Brier score: "
    f"{test_results['XGBoost']['Brier Score']:.4f}"
)
report.append(
    "A lower Brier score is better. The calibration curve should also be "
    "examined to assess whether predicted probabilities match observed "
    "failure frequencies."
)
report.append("")
report.append("## 7. Limitations")
report.append(
    "- The dataset is imbalanced, with fewer failure cases than normal cases."
)
report.append(
    "- Results on the UCI dataset may not generalize to every real-world machine."
)
report.append(
    "- The test set has already been used for earlier model comparisons; "
    "it should not be described as a completely untouched final evaluation."
)
report.append(
    "- Probability calibration and maintenance thresholds require further "
    "validation before real industrial use."
)
report.append("")
report.append("## 8. Conclusion")
report.append(
    "Both models show promising predictive performance on the AI4I dataset. "
    "XGBoost is the current leading candidate based on cross-validation, "
    "but real-world deployment requires additional validation."
)

# Save report
output_dir = BASE_DIR / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "model_evaluation_report.md"
output_path.write_text("\n".join(report), encoding="utf-8")

print("\nMODEL EVALUATION REPORT GENERATED")
print("=" * 45)
print(f"Report saved at: {output_path}")
print("\nSelected model: XGBoost")
print("Report includes cross-validation, test metrics,")
print("calibration score, model selection and limitations.")
