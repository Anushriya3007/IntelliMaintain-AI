
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cross-validation results
results_path = BASE_DIR / "data" / "processed" / "model_validation_cv.csv"
df = pd.read_csv(results_path)

print("\nINTELLIMAINTAIN-AI MODEL COMPARISON")
print("=" * 55)

# Display selected metrics
metrics = ["PRECISION", "RECALL", "F1", "ROC_AUC", "PR_AUC"]

print(df[["Model"] + [f"{m} Mean" for m in metrics]].to_string(index=False))

# Prepare chart
x = np.arange(len(metrics))
width = 0.35

rf_values = [
    df.loc[df["Model"] == "Random Forest", f"{m} Mean"].iloc[0]
    for m in metrics
]

xgb_values = [
    df.loc[df["Model"] == "XGBoost", f"{m} Mean"].iloc[0]
    for m in metrics
]

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(x - width / 2, rf_values, width, label="Random Forest")
ax.bar(x + width / 2, xgb_values, width, label="XGBoost")

ax.set_title("5-Fold Cross-Validation: Model Performance")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save chart
output_path = BASE_DIR / "models" / "explanations" / "model_comparison.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nComparison chart saved to:")
print(output_path)
