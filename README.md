# Credit Card Default Prediction

## 1. Project Overview

This project presents an end-to-end data science solution for predicting credit card default. The primary goal is to build a reliable machine learning model that can identify clients likely to default on their payments in the upcoming month. By leveraging a historical dataset, this project demonstrates a complete workflow, including:

*   In-depth Exploratory Data Analysis (EDA) to uncover patterns and insights.
*   Meaningful Feature Engineering to create powerful, business-driven predictors.
*   A robust Preprocessing Pipeline using `sklearn.Pipeline` and `ColumnTransformer`.
*   Comparative analysis of multiple baseline models using K-Fold Cross-Validation.
*   Hyperparameter Tuning of the champion model (XGBoost) using `Optuna`.
*   In-depth model evaluation and threshold calibration to align with business goals.

The final model provides a powerful and actionable tool for proactive risk management, enabling financial institutions to minimize credit losses.

---

## 2. Business Problem

Credit default poses a significant financial risk to lending institutions. This project aims to mitigate this risk by developing a predictive model. The model's insights can be used to:
- Implement targeted, proactive interventions for high-risk clients.
- Optimize debt collection strategies.
- Make more informed, data-driven decisions on credit line management.

---

## 3. Dataset

The dataset used is the "Default of Credit Card Clients Dataset" from the UCI Machine Learning Repository. It contains demographic and historical payment data for 30,000 credit card clients in Taiwan from April to September 2005.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 4. Methodology

The project follows a structured data science lifecycle:

1.  **Exploratory Data Analysis (EDA):** Investigated data distributions, identified correlations, and analyzed the relationship between features and the target variable (default). Key findings include a significant class imbalance (22% defaulters) and the high predictive power of recent payment history.

2.  **Feature Engineering:** Created over 10 new features to better capture client behavior, such as:
    *   `AVG_UTILIZATION`: Average credit limit utilization over 6 months.
    *   `AVG_PAY_STATUS`: Average payment status score, summarizing payment discipline.
    *   `NUM_MONTHS_LATE`: A simple counter for the number of delayed payments.

3.  **Feature Selection:** Used `RandomForestClassifier`'s feature importance to identify and remove redundant original features, reducing model complexity and multicollinearity.

4.  **Modeling & Validation:**
    *   A robust preprocessing pipeline was built using `sklearn.Pipeline` and `ColumnTransformer` to handle missing values, encode categorical features, and scale numerical data.
    *   Several baseline models (Logistic Regression, Random Forest, XGBoost) were evaluated using **5-fold stratified cross-validation**.
    *   The best-performing baseline model was selected for further optimization.

5.  **Hyperparameter Tuning:** The champion model was fine-tuned using **Optuna**, a modern Bayesian optimization framework, to find the optimal set of hyperparameters that maximized the F1-score.

---

## 5. Final Results & Business Impact

The final optimized model was trained on the full training set and evaluated on the unseen test set.

#### Key Performance Metrics:

After calibrating the prediction threshold to optimize the F1-score for the minority class, the model achieved:

*   **AUC-ROC:** **0.77** (Good discriminative ability)
*   **Recall (for 'Default' class):** **59%** (Successfully identifies 59% of all actual defaulters)
*   **Precision (for 'Default' class):** **49%** (When predicting a default, the model is correct 49% of the time)
*   **Weighted F1-Score:** **0.78**

#### Economic Impact Simulation:

To translate these metrics into business value, a simple economic simulation was performed on the test set (~6,000 clients). Assuming a hypothetical average loss of **$2,000** for a missed defaulter (False Negative) and an opportunity cost of **$200** for a wrongly flagged good client (False Positive), the model demonstrates:

*   **Potential Savings:** **~$1,352,400** on this cohort of clients.
*   **Reduction in Credit Loss:** Approximately **51%**.

This simulation clearly shows that the model, despite its imperfections, provides a massive positive financial impact by enabling proactive risk mitigation.
