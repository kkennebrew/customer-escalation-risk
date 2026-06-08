# Customer Service Escalation Risk Modelling

A multi-model classification and clustering project that predicts customer service escalation risk levels from interaction data, then validates those risk groupings using unsupervised learning.

## Overview

Using a dataset of customer service interactions, this project builds and compares four classification models to predict whether a customer interaction will escalate. A secondary clustering analysis is then applied to discover natural customer groupings and verify their alignment with escalation behaviour, without using the target label.

## Models built

| Model | Approach |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable rule-based model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting for higher accuracy |
| K-Means Clustering | Unsupervised grouping to validate segments |

## Key findings

- Gradient Boosting achieved the highest classification accuracy and F1 score across escalation levels
- Customer sentiment score and engagement level were the strongest predictors of escalation risk
- Clustering revealed three natural customer segments that align closely with low, medium, and high escalation groups, confirming the feature engineering approach
- Class imbalance in the target variable required careful handling during preprocessing

## Tools and technologies

| Tool | Purpose |
|---|---|
| Python | Full pipeline |
| pandas | Data cleaning, encoding, feature engineering |
| scikit-learn | Model training, evaluation, and clustering |
| rapidfuzz / thefuzz | Fuzzy string matching for categorical cleaning |
| Matplotlib / Seaborn / Missingno | Visualisations and missingness analysis |
| Jupyter Notebook | Analysis environment |

## Files

| File | Description |
|---|---|
| `Advanced Data Analysis.ipynb` | Full notebook — EDA, preprocessing, modelling, clustering |
| `customer_service_escalation_risk.csv` | Raw dataset |
| `Advanced Data Analysis Report.pdf` | Written report |

## Running the notebook

```bash
pip install pandas scikit-learn matplotlib seaborn missingno rapidfuzz
jupyter notebook "Advanced Data Analysis.ipynb"
```

---
*Data Mining module — Coursework 2, 2025–26.*
