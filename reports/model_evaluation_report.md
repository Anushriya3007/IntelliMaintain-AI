# IntelliMaintain-AI: Model Evaluation Report

## 1. Project Overview
IntelliMaintain-AI is a machine learning-based predictive maintenance system designed to identify potential machine failures using operational sensor measurements.

## 2. Dataset
- Dataset: UCI AI4I 2020 Predictive Maintenance Dataset
- Test samples: 2000
- Training samples: 8000
- Failure cases in test set: 68
- Normal cases in test set: 1932

## 3. Five-Fold Cross-Validation
| Model         |   PRECISION Mean |   PRECISION Std |   RECALL Mean |   RECALL Std |   F1 Mean |    F1 Std |   ROC_AUC Mean |   ROC_AUC Std |   PR_AUC Mean |   PR_AUC Std |
|:--------------|-----------------:|----------------:|--------------:|-------------:|----------:|----------:|---------------:|--------------:|--------------:|-------------:|
| Random Forest |         0.541551 |       0.0346441 |      0.767542 |    0.0144767 |  0.634204 | 0.0224356 |       0.970851 |     0.0118818 |      0.706518 |    0.0368371 |
| XGBoost       |         0.697719 |       0.0368321 |      0.767677 |    0.0369911 |  0.730043 | 0.0263895 |       0.970518 |     0.0111657 |      0.769633 |    0.0365138 |

## 4. Test-Set Performance
### Random Forest
- **Accuracy:** 0.97
- **Precision:** 0.5408163265306123
- **Recall:** 0.7794117647058824
- **F1-score:** 0.6385542168674698
- **ROC-AUC:** 0.970017354767994
- **PR-AUC:** 0.7348738043197702
- **Brier Score:** 0.023914920425750365
- **Confusion Matrix:** [[1887, 45], [15, 53]]

### XGBoost
- **Accuracy:** 0.979
- **Precision:** 0.6547619047619048
- **Recall:** 0.8088235294117647
- **F1-score:** 0.7236842105263158
- **ROC-AUC:** 0.974751857264645
- **PR-AUC:** 0.8391232876676864
- **Brier Score:** 0.015209957025945187
- **Confusion Matrix:** [[1903, 29], [13, 55]]

## 5. Model Selection
XGBoost is the current preferred model because it achieved stronger cross-validation precision, F1-score and PR-AUC than Random Forest. Final deployment decisions should also consider false negatives, false positives and operational requirements.

## 6. Probability Calibration
- XGBoost test-set Brier score: 0.0152
A lower Brier score is better. The calibration curve should also be examined to assess whether predicted probabilities match observed failure frequencies.

## 7. Limitations
- The dataset is imbalanced, with fewer failure cases than normal cases.
- Results on the UCI dataset may not generalize to every real-world machine.
- The test set has already been used for earlier model comparisons; it should not be described as a completely untouched final evaluation.
- Probability calibration and maintenance thresholds require further validation before real industrial use.

## 8. Conclusion
Both models show promising predictive performance on the AI4I dataset. XGBoost is the current leading candidate based on cross-validation, but real-world deployment requires additional validation.