# Eskom Power Station Failure Prediction

### Digital Framework for Predicting Plant Failures among South African Power Stations using Machine Learning

---

## 📌 Project Overview
This repository hosts a multi-stage machine learning pipeline developed to predict plant failures across six Eskom coal-fired power stations. Utilizing load loss event data spanning five financial years (FY2018–FY2022), this framework addresses failure event identification, severity forecasting, location/cause prediction, anomaly detection, and long-term risk ranking.

### ⚡ Key Prediction Targets:
1. **Classification:** Predict whether a capability loss event constitutes an unplanned failure (UCLF flag).
2. **Regression:** Forecast the severity of failure measured in Megawatts lost (`MWLOSS`).
3. **Multi-Output Location:** Simultaneously predict which Station, Plant Area, and Cause Category are responsible.
4. **Unsupervised Anomaly Detection:** Detect operational deviations to issue early warnings.
5. **Prophet Time-Series Forecasting:** Predict weekly failure counts over an 8-to-16-week horizon.

---

## 📊 Data Summary
- **Source:** Eskom Load Loss Event Records.
- **Period:** FY2018 – FY2022 (April 2017 – March 2022).
- **Observations:** 65,637 cleaned events across the top 6 coal-fired power stations:
  - *Arnot, Camden, Duvha, Kendal, Lethabo, Medupi*
- **Feature Engineering:** Failure indicators, seasonal metrics, plant unique keys, rolling 3-month failure histories, and survival durations.

---

## 🚀 Pipeline & Key Findings

### 1. Binary Classification (Supervised Early Warning)
To mitigate the massive class imbalance, **SMOTE** oversampling was applied.
- **Top Models:** *XGBoost, LightGBM, and Random Forest*.
- **Performance:** Achieved **ROC-AUC > 0.99** and **F1-Score > 0.98**.
- **Operational Early Warning:** At a 70% probability threshold, the **XGBoost** classifier caught **99.1% of true failures** with a **false alarm rate of only 0.9%**.
- **Primary Drivers:** Primary failure cause (`CAUSE_DESCRIP1_ENC`) and rolling 3-month failure histories (`ROLLING_FAILURES_3M`).

### 2. Regression (Predicting Severity)
Evaluating 9 models to predict Megawatts lost (`MWLOSS`) exclusively on true unplanned (UCLF) failures:
- **Best Model:** **Random Forest Regressor** achieved an **R² score of 0.9936** with a **Mean Absolute Error (MAE) of 2.37 MW**.
- **Actionable Banding:** Predicted severity is categorized into dynamic bands (*Low, Moderate, High, Critical*) to streamline dispatch priorities.

### 3. Multi-Output Prediction (Spatio-Temporal Tracking)
Simultaneous multiclass prediction across three operational vectors using chained estimators:
- **Top Model:** **Chained XGBoost** achieved an **Average Joint Accuracy of 94.48%**.
  - *Plant Area Prediction Accuracy:* **99.73%**
  - *Cause Category Prediction Accuracy:* **99.43%**
  - *Station Prediction Accuracy:* **84.28%**

### 4. Unsupervised Anomaly Detection
Designed to flag abnormal operations before they cascade into system-wide failures. Models compared: *Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM*.
- **Best Model:** **One-Class SVM** achieved **97.00% precision** for identifying true anomalies.
- **Consensus Alerts:** Out of 468 consensus alerts (flagged by all 3 models), **87.6%** were verified true failures.
- **Station Hotspots:** Medupi accounted for **53.2%** of all high-confidence system anomalies.

### 5. Prophet Time-Series Forecasting (Strategic Asset Scheduling)
Prophet models forecast the failure landscape 8 to 16 weeks ahead based on yearly seasonality patterns.
- **4-Month Strategic Risk Ranking (Critical to Low):**
  1. 🔴 **Arnot** (235 forecasted failures)
  2. 🟠 **Duvha** (193 forecasted failures)
  3. 🟡 **Medupi** (166 forecasted failures)
  4. 🟢 **Kendal** (163) | **Camden** (131) | **Lethabo** (45)

---

## 🛠️ Replication & Implementation

1. **Environment Setup:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn prophet xgboost lightgbm imbalanced-learn
   ```

2. **Run the Notebook:**
   Execute `Eskom_Power_Station_Failure_Prediction.ipynb` sequentially to perform data loading, cleaning, model training, and asset ranking.

3. **Inspect Output Artifacts:**
   The model exports predictions and configuration files directly to disk:
   - `04_model_comparison.csv` (Classifier comparison metrics)
   - `04_early_warning_flags.csv` (High-priority failure alarms)
   - `04b_plant_clusters.csv` (K-Means risk groupings)
   - `05a_mwloss_predictions.csv` (Severity regression forecasts)
   - `06_maintenance_priority.csv` (Prophet resource-allocation ranking)

---
