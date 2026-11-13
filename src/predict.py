
# IntelliMaintain-AI
# Step 38: Prediction Module

import joblib
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load trained models
rf_model = joblib.load(
    MODELS_DIR / "random_forest.pkl"
)

xgb_model = joblib.load(
    MODELS_DIR / "xgboost_model.pkl"
)

preprocessor = joblib.load(
    MODELS_DIR / "preprocessor.pkl"
)


def predict_failure(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    # Create input dataframe
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])
    # XGBoost prediction
    # The XGBoost pipeline handles preprocessing internally
    prediction = xgb_model.predict(input_data)[0]
    probability = xgb_model.predict_proba(input_data)[0][1]
    return {
        "prediction": int(prediction),
        "status": "FAILURE" if prediction == 1 else "NORMAL",
        "failure_probability": round(float(probability) * 100, 2)
    }


print("IntelliMaintain-AI prediction module loaded!")
