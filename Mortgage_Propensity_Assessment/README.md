
# Mortgage Propensity Assessment

## Problem Statement

A bank has provided two datasets: historical **retail customer data** with known mortgage outcomes, and a list of **potential new customers**. The goal was to:

- Build a predictive model to identify customers likely to take a **mortgage**
- Apply this model to new prospects
- Deliver a shortlist of high-confidence, actionable leads

---

## Dataset Overview

### 1. `retail_df.csv`
- Contains 23,983 customers
- Target variable: `mortgage_yn` (Yes/No)
- Only **1.3%** of customers have a mortgage → strong class imbalance

### 2. `potential_df.csv`
- 2,747 new potential customers
- Same features, no target
- Model inference applied here

---

## Data Preprocessing

### Key Steps:
- Removed or capped extreme outliers (`cust_income`, `address_stability_years`)
- Created log-transformed `cust_income_log` for normalization
- Extracted job/address stability features from dates
- Imputed missing values:
  - Used model-based imputation for `marital_status`
  - Added binary flags for missing stability metrics
- Ensured all preprocessing steps were applied identically to both datasets

---

## Feature Engineering

Final features used in modeling were selected via **Recursive Feature Elimination (RFE)**:
- `age`
- `years_with_bank`
- `marital_status_M`
- `employment_PVE`
- `gender_M`
- `job_stability_years`
- `address_stability_years`
- `cust_income_log`

Categorical features were encoded using `OneHotEncoder`, and all features were scaled with `StandardScaler`.

---

## Modeling & Evaluation

Three models were compared using **Stratified K-Fold Cross-Validation**:

| Model         | ROC AUC | PR AUC |
|---------------|---------|--------|
| RandomForest  | 0.814   | 0.063  |
| XGBoost       | 0.754   | 0.049  |
| **CatBoost**  | **0.829** | **0.077** |

### Why PR AUC is Low
- The positive class prevalence is just **1.3%**
- The baseline PR AUC (random guess) is ~0.013
- CatBoost significantly outperforms this baseline

---

## 🔧 Calibration & Threshold Optimization

To improve real-world utility, we **calibrated the predicted probabilities** using **isotonic regression**. This made scores interpretable (e.g. a 0.30 score ≈ 30% actual chance) and improved F1 performance.

| Metric     | Value |
|------------|-------|
| **Calibrated Threshold**  | **0.103** |
| **Precision**  | 18.0% |
| **Recall**     | ~33% |
| **F1 Score**   | 0.20 |
| **PR AUC**     | 0.108 |

---

## 🔍 Inference on Potential Customers

- The calibrated CatBoost model was applied to `potential_df`
- Using the new threshold, **56 out of 2,747** customers were flagged as **likely mortgage prospects**
- These leads are sorted by calibrated probability and ready for CRM outreach

Output file: `output/potential_df_scored.csv`

---

## 📁 Project Structure
```
|-- README.md
|-- data
|   |-- Potential Customers.csv
|   `-- Retail data.csv
|-- models
|   `-- encoder_retail.pkl
|-- notebook
|   `-- notebook.ipynb
|-- output
|   `-- potential_df_scored.csv
|-- presentation
|-- reports
|   |-- Compare.html
|   |-- Potential Profile.html
|   |-- Retail Profile 2.html
|   `-- Retail Profile.html
`-- requirements.txt
```

---

## Key Takeaways

- Handled data quality, imbalance, and modeling from end to end
- Applied **probability calibration** for better interpretability and decision-making
- Tuned threshold to **maximize F1 score**, balancing precision and recall
- Delivered a **shortlist of 56 actionable leads** for targeted CRM follow-up

---

## Author
**Konstantinos Soufleros**  
Certified Data Scientist | [GitHub](https://github.com/kostas696)
