from datetime import datetime
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_failure


st.set_page_config(
    page_title="IntelliMaintain-AI",
    page_icon="⚙️",
    layout="wide"
)

HISTORY_FILE = BASE_DIR / "data" / "processed" / "prediction_history.csv"
MODEL_FILE = BASE_DIR / "models" / "xgboost_model.pkl"


# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def classify_machine_risk(probability_percent):
    """Classify the model's predicted failure probability."""
    if probability_percent < 30:
        return "LOW", "🟢", "Machine can continue normal operation."
    elif probability_percent < 70:
        return "MEDIUM", "🟡", "Schedule a maintenance inspection."
    return "HIGH", "🔴", "Prioritize machine inspection."


def save_prediction_history(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    prediction,
    probability
):
    """Append one prediction to the CSV history file."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = pd.DataFrame([{
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Machine Type": machine_type,
        "Air Temperature (K)": air_temperature,
        "Process Temperature (K)": process_temperature,
        "Rotational Speed (rpm)": rotational_speed,
        "Torque (Nm)": torque,
        "Tool Wear (min)": tool_wear,
        "Prediction": int(prediction),
        "Status": "FAILURE" if int(prediction) == 1 else "NORMAL",
        "Failure Probability (%)": float(probability)
    }])

    record.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not HISTORY_FILE.exists(),
        index=False
    )


@st.cache_resource
def load_xgb_model():
    """Load the complete fitted XGBoost preprocessing pipeline."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Please make sure models/xgboost_model.pkl exists."
        )
    return joblib.load(MODEL_FILE)


def get_maintenance_recommendations(
    risk_level,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    """Create simple rule-based suggestions from the entered readings."""
    recommendations = []

    temperature_difference = process_temperature - air_temperature

    if risk_level == "HIGH":
        recommendations.append(
            "Arrange an inspection before continuing normal operation."
        )
    elif risk_level == "MEDIUM":
        recommendations.append(
            "Schedule a maintenance inspection and monitor the next readings."
        )
    else:
        recommendations.append(
            "Continue routine monitoring and follow the normal maintenance schedule."
        )

    if temperature_difference < 8:
        recommendations.append(
            "The air-to-process temperature difference is relatively small; "
            "check temperature sensors and cooling conditions."
        )

    if rotational_speed >= 2500:
        recommendations.append(
            "Rotational speed is high; inspect operating conditions and vibration."
        )

    if torque >= 60:
        recommendations.append(
            "Torque is elevated; inspect load conditions and the drive system."
        )

    if tool_wear >= 200:
        recommendations.append(
            "Tool wear is high; plan a tool-condition check or replacement."
        )

    if len(recommendations) == 1:
        recommendations.append(
            "No additional rule-based warning was triggered by the entered readings."
        )

    return recommendations


def load_history():
    """Read prediction history safely."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    try:
        history = pd.read_csv(HISTORY_FILE)
        required_columns = {
            "Date & Time",
            "Machine Type",
            "Status",
            "Failure Probability (%)"
        }
        if not required_columns.issubset(history.columns):
            st.warning(
                "The prediction history file does not contain all expected columns."
            )
            return pd.DataFrame()
        return history
    except Exception as error:
        st.warning(f"Could not read prediction history: {error}")
        return pd.DataFrame()


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
    }

    h1, h2, h3, p, label {
        color: white;
    }

    .normal-box {
        background-color: #14532d;
        padding: 20px;
        border-radius: 12px;
        color: white;
    }

    .failure-box {
        background-color: #991b1b;
        padding: 20px;
        border-radius: 12px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 4. HEADER
# =====================================================

st.title("⚙️ IntelliMaintain-AI")
st.subheader("AI-Powered Predictive Maintenance System")
st.caption("Machine Learning | XGBoost | Random Forest | SHAP")
st.divider()


# =====================================================
# 5. SIDEBAR - MACHINE CONFIGURATION
# =====================================================

st.sidebar.title("Machine Configuration")
st.sidebar.caption("Enter the current machine sensor readings.")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=500,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)


# =====================================================
# 6. MACHINE MONITORING DASHBOARD
# =====================================================

st.subheader("Machine Monitoring Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Air Temperature", f"{air_temperature:.1f} K")

with col2:
    st.metric("Process Temperature", f"{process_temperature:.1f} K")

with col3:
    st.metric("Rotational Speed", f"{rotational_speed} rpm")

st.divider()


# =====================================================
# 7. PREDICTION
# =====================================================

if st.button("🔍 Predict Machine Failure", use_container_width=True):

    try:
        # The prediction function accepts the original six input values.
        # Its saved pipeline performs preprocessing internally.
        result = predict_failure(
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear
        )

        prediction = int(result["prediction"])
        probability = float(result["failure_probability"])
        probability = min(max(probability, 0.0), 100.0)

        risk_level, risk_icon, risk_message = classify_machine_risk(probability)

        save_prediction_history(
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            prediction,
            probability
        )

        st.success("Prediction completed and saved to history.")

        # -------------------------------------------------
        # 8. PREDICTION RESULTS
        # -------------------------------------------------

        st.subheader("Prediction Results")
        st.subheader("🏭 Machine Health Assessment")

        if risk_level == "LOW":
            st.success(f"{risk_icon} LOW RISK — {risk_message}")
        elif risk_level == "MEDIUM":
            st.warning(f"{risk_icon} MEDIUM RISK — {risk_message}")
        else:
            st.error(f"{risk_icon} HIGH RISK — {risk_message}")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 0:
                st.markdown(
                    """
                    <div class="normal-box">
                        <h2>🟢 MACHINE NORMAL</h2>
                        <p>No failure predicted by the model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="failure-box">
                        <h2>🔴 MACHINE FAILURE</h2>
                        <p>Potential failure detected by the model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with probability_col:
            st.metric("Failure Probability", f"{probability:.2f}%")
            st.progress(probability / 100.0)

        # -------------------------------------------------
        # 9. FAILURE PROBABILITY GAUGE
        # -------------------------------------------------

        st.divider()
        st.subheader("Failure Probability Gauge")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={"suffix": "%", "valueformat": ".2f"},
                title={"text": "Predicted Failure Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#EF4444"},
                    "steps": [
                        {"range": [0, 30], "color": "#166534"},
                        {"range": [30, 70], "color": "#A16207"},
                        {"range": [70, 100], "color": "#7F1D1D"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": probability
                    }
                }
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            height=350,
            margin={"t": 70, "b": 20, "l": 30, "r": 30}
        )

        st.plotly_chart(gauge, use_container_width=True)

        # -------------------------------------------------
        # 10. SMART MAINTENANCE RECOMMENDATIONS
        # -------------------------------------------------

        st.divider()
        st.subheader("🛠️ Smart Maintenance Recommendations")

        recommendations = get_maintenance_recommendations(
            risk_level,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear
        )

        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

        st.caption(
            "These are rule-based suggestions using the entered readings "
            "and model risk category. They are not a substitute for a "
            "qualified maintenance inspection."
        )

        # -------------------------------------------------
        # 11. SHAP EXPLAINABILITY
        # -------------------------------------------------

        st.divider()
        st.subheader("🔍 SHAP - Prediction Explainability")

        try:
            with st.spinner("Calculating SHAP feature contributions..."):

                # Use the same raw input columns and order as training.
                input_data = pd.DataFrame([{
                    "Type": machine_type,
                    "Air temperature [K]": air_temperature,
                    "Process temperature [K]": process_temperature,
                    "Rotational speed [rpm]": rotational_speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": tool_wear
                }])

                xgb_pipeline = load_xgb_model()
                preprocessor = xgb_pipeline.named_steps["preprocessor"]
                classifier = xgb_pipeline.named_steps["classifier"]

                transformed_input = preprocessor.transform(input_data)

                if hasattr(transformed_input, "toarray"):
                    transformed_input = transformed_input.toarray()

                transformed_input = np.asarray(transformed_input)
                feature_names = list(preprocessor.get_feature_names_out())

                explainer = shap.TreeExplainer(classifier)
                shap_result = explainer(transformed_input)

                shap_array = np.asarray(shap_result.values)
                base_array = np.asarray(shap_result.base_values)

                # Handle binary classifier SHAP output formats.
                if shap_array.ndim == 3:
                    sample_shap_values = shap_array[0, :, 1]
                    if base_array.ndim == 2:
                        sample_base_value = float(base_array[0, 1])
                    else:
                        sample_base_value = float(base_array.reshape(-1)[1])
                else:
                    sample_shap_values = shap_array[0]
                    sample_base_value = float(base_array.reshape(-1)[0])

                if len(sample_shap_values) != len(feature_names):
                    raise ValueError(
                        "SHAP feature count does not match the preprocessed feature names."
                    )

                # -------------------------------------------------
                # 12. SHAP CONTRIBUTION BAR CHART
                # -------------------------------------------------

                st.markdown("### Sensor Contributions to the Prediction")

                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Contribution": sample_shap_values
                })
                shap_df["Absolute Contribution"] = (
                    shap_df["SHAP Contribution"].abs()
                )
                shap_df = shap_df.sort_values(
                    "Absolute Contribution",
                    ascending=True
                )

                shap_fig = go.Figure()
                shap_fig.add_trace(
                    go.Bar(
                        x=shap_df["SHAP Contribution"],
                        y=shap_df["Feature"],
                        orientation="h",
                        marker_color=[
                            "#EF4444" if value > 0 else "#22C55E"
                            for value in shap_df["SHAP Contribution"]
                        ],
                        hovertemplate=(
                            "%{y}<br>SHAP contribution: %{x:.4f}"
                            "<extra></extra>"
                        )
                    )
                )
                shap_fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="white"
                )
                shap_fig.update_layout(
                    title="Feature Contributions for This Prediction",
                    xaxis_title="SHAP contribution (model output scale)",
                    yaxis_title="Machine features",
                    template="plotly_dark",
                    height=450
                )
                st.plotly_chart(shap_fig, use_container_width=True)

                # -------------------------------------------------
                # 13. INDIVIDUAL SHAP WATERFALL
                # -------------------------------------------------

                st.divider()
                st.subheader("📊 Individual Prediction Explanation")

                waterfall_explanation = shap.Explanation(
                    values=sample_shap_values,
                    base_values=sample_base_value,
                    data=transformed_input[0],
                    feature_names=feature_names
                )

                waterfall_fig = plt.figure(figsize=(11, 6))
                shap.plots.waterfall(
                    waterfall_explanation,
                    max_display=10,
                    show=False
                )
                plt.title("How Each Sensor Influenced This Prediction")
                plt.tight_layout()
                st.pyplot(waterfall_fig, clear_figure=True)
                plt.close(waterfall_fig)

                # -------------------------------------------------
                # 14. SHAP INTERPRETATION
                # -------------------------------------------------

                st.markdown("### How to Interpret SHAP")
                st.markdown(
                    """
                    - 🔴 **Red features:** Push the model output higher.
                    - 🔵 **Blue features:** Push the model output lower
                      in the waterfall plot.
                    - Larger absolute contributions indicate a stronger
                      influence on this individual model prediction.
                    - SHAP explains model behaviour; it does not prove
                      that a sensor reading causes machine failure.
                    """
                )

                top_features = shap_df.sort_values(
                    "Absolute Contribution",
                    ascending=False
                ).head(3)

                st.markdown("### Top 3 Influential Features")

                for _, row in top_features.iterrows():
                    direction = (
                        "Pushes model output higher"
                        if row["SHAP Contribution"] > 0
                        else "Pushes model output lower"
                    )
                    st.write(f"**{row['Feature']}** — {direction}")

        except Exception as shap_error:
            st.warning(
                "The prediction was successful, but the SHAP explanation "
                f"could not be generated: {shap_error}"
            )

    except Exception as error:
        st.error(f"Prediction error: {error}")


# =====================================================
# 15. MACHINE PREDICTION HISTORY
# =====================================================

st.divider()
st.subheader("📜 Machine Prediction History")

history_df = load_history()

if not history_df.empty:
    st.metric("Total Predictions", len(history_df))

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="IntelliMaintain_Prediction_History.csv",
        mime="text/csv"
    )

    normal_count = int((history_df["Status"] == "NORMAL").sum())
    failure_count = int((history_df["Status"] == "FAILURE").sum())

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric("Normal Predictions", normal_count)

    with summary_col2:
        st.metric("Failure Predictions", failure_count)

else:
    st.info("No prediction history found. Run a machine prediction to create history.")


# =====================================================
# 16. PREDICTION ANALYTICS DASHBOARD
# =====================================================

st.divider()
st.subheader("📊 Prediction Analytics Dashboard")

if not history_df.empty:
    analytics_df = history_df.copy()

    total_predictions = len(analytics_df)
    normal_count = int((analytics_df["Status"] == "NORMAL").sum())
    failure_count = int((analytics_df["Status"] == "FAILURE").sum())
    failure_percentage = (
        (failure_count / total_predictions) * 100
        if total_predictions
        else 0.0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total Predictions", total_predictions)
    metric2.metric("Normal Predictions", normal_count)
    metric3.metric("Failure Predictions", failure_count)
    metric4.metric("Failure Percentage", f"{failure_percentage:.2f}%")

    # -------------------------------------------------
    # 16A. NORMAL VS FAILURE DISTRIBUTION
    # -------------------------------------------------

    st.markdown("### 1. Normal vs Failure Distribution")

    pie_fig = go.Figure(
        go.Pie(
            labels=["Normal", "Failure"],
            values=[normal_count, failure_count],
            hole=0.5,
            marker={"colors": ["#22C55E", "#EF4444"]},
            textinfo="label+percent"
        )
    )
    pie_fig.update_layout(
        title="Machine Prediction Distribution",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # -------------------------------------------------
    # 16B. FAILURE PROBABILITY TREND
    # -------------------------------------------------

    st.markdown("### 2. Failure Probability Trend")

    analytics_df["Date & Time"] = pd.to_datetime(
        analytics_df["Date & Time"],
        errors="coerce"
    )
    trend_df = analytics_df.dropna(
        subset=["Date & Time"]
    ).sort_values("Date & Time")

    if not trend_df.empty:
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=trend_df["Date & Time"],
                y=trend_df["Failure Probability (%)"],
                mode="lines+markers",
                name="Failure Probability",
                line={"color": "#38BDF8", "width": 3},
                marker={"size": 8}
            )
        )
        trend_fig.update_layout(
            title="Failure Probability Over Prediction Time",
            xaxis_title="Prediction Time",
            yaxis_title="Failure Probability (%)",
            yaxis={"range": [0, 100]},
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("No valid timestamps are available for the trend chart.")

    # -------------------------------------------------
    # 16C. MACHINE TYPE ANALYSIS
    # -------------------------------------------------

    st.markdown("### 3. Machine Type-wise Analysis")

    machine_summary = (
        analytics_df.groupby(["Machine Type", "Status"])
        .size()
        .unstack(fill_value=0)
    )

    machine_fig = go.Figure()

    for status, color in [
        ("NORMAL", "#22C55E"),
        ("FAILURE", "#EF4444")
    ]:
        if status in machine_summary.columns:
            machine_fig.add_trace(
                go.Bar(
                    x=machine_summary.index,
                    y=machine_summary[status],
                    name=status,
                    marker_color=color
                )
            )

    machine_fig.update_layout(
        title="Predictions by Machine Type",
        xaxis_title="Machine Type",
        yaxis_title="Number of Predictions",
        barmode="group",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(machine_fig, use_container_width=True)

else:
    st.info("Generate machine predictions to view analytics.")


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()
st.caption("IntelliMaintain-AI | Predictive Maintenance using Machine Learning")

# =====================================================
# 17. MODEL PERFORMANCE DASHBOARD
# =====================================================

st.divider()
st.subheader("📈 Model Performance & Evaluation")

# Load cross-validation results
CV_FILE = BASE_DIR / "data" / "processed" / "model_validation_cv.csv"

if CV_FILE.exists():

    cv_results = pd.read_csv(CV_FILE)

    st.markdown("### 1. Five-Fold Cross-Validation Results")

    display_columns = [
        "Model",
        "PRECISION Mean",
        "RECALL Mean",
        "F1 Mean",
        "ROC_AUC Mean",
        "PR_AUC Mean"
    ]

    st.dataframe(
        cv_results[display_columns].round(4),
        use_container_width=True,
        hide_index=True
    )

    # Model comparison chart
    st.markdown("### 2. Random Forest vs XGBoost")

    metrics = ["PRECISION", "RECALL", "F1", "ROC_AUC", "PR_AUC"]

    comparison_fig = go.Figure()

    for model_name in ["Random Forest", "XGBoost"]:

        model_row = cv_results[
            cv_results["Model"] == model_name
        ]

        if not model_row.empty:

            comparison_fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=metrics,
                    y=[
                        model_row[f"{metric} Mean"].iloc[0]
                        for metric in metrics
                    ]
                )
            )

    comparison_fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Evaluation Metrics",
        yaxis_title="Cross-Validation Score",
        yaxis={"range": [0, 1.05]},
        barmode="group",
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(comparison_fig, use_container_width=True)

    # Selected model
    st.markdown("### 3. Selected Model")

    st.success(
        "XGBoost is the current preferred model based on "
        "its cross-validation F1-score, precision and PR-AUC."
    )

    # Calibration evaluation
    st.markdown("### 4. Probability Calibration")

    calibration_file = (
        BASE_DIR / "models" / "explanations" / "calibration_curve.png"
    )

    if calibration_file.exists():

        st.image(
            str(calibration_file),
            caption="XGBoost Probability Calibration Curve",
            use_container_width=True
        )

        st.metric("XGBoost Test Brier Score", "0.0152")

        st.caption(
            "A lower Brier score is better. The calibration curve "
            "should also be examined to assess probability reliability."
        )

    else:
        st.info("Calibration chart not found. Run calibrate_model.py.")

    # Evaluation report
    st.markdown("### 5. Detailed Evaluation Report")

    report_file = BASE_DIR / "reports" / "model_evaluation_report.md"

    if report_file.exists():

        with st.expander("View Complete Model Evaluation Report"):

            report_text = report_file.read_text(encoding="utf-8")
            st.markdown(report_text)

    else:
        st.info("Model evaluation report not found.")

else:

    st.warning(
        "Cross-validation results not found. "
        "Run python .\\src\\validate.py first."
    )
