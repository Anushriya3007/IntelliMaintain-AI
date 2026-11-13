
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load test dataset
test_path = BASE_DIR / "data" / "processed" / "test.csv"
df = pd.read_csv(test_path)

X_test = df.drop(columns=["Machine failure"])
y_test = df["Machine failure"]

# Load XGBoost pipeline
model_path = BASE_DIR / "models" / "xgboost_model.pkl"
model = joblib.load(model_path)

# Get predicted failure probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate Brier score
brier = brier_score_loss(y_test, y_prob)

# Calculate calibration curve
actual_rate, predicted_mean = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="quantile"
)

print("\nINTELLIMAINTAIN-AI PROBABILITY CALIBRATION")
print("=" * 50)
print(f"Brier Score: {brier:.4f}")
print("(Lower Brier score is better; 0 is perfect.)")

# Plot calibration curve
plt.figure(figsize=(8, 6))

plt.plot(
    predicted_mean,
    actual_rate,
    marker="o",
    label="XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Failure Probability")
plt.ylabel("Actual Failure Rate")
plt.title("XGBoost Probability Calibration")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save chart
output_path = (
    BASE_DIR / "models" / "explanations" / "calibration_curve.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nCalibration chart saved to:")
print(output_path)
