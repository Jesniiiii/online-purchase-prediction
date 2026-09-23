# Online Purchase Prediction

Predicts whether an online shopping session will result in a purchase, based on session behavior and browsing patterns. Built as a classification task using a mandatory ensemble of **Random Forest**, **XGBoost**, and **CatBoost**.

## Problem Statement

E-commerce platforms want to identify, in real time, which visitor sessions are likely to convert into a purchase. This project analyzes customer behavior during a browsing session and predicts purchase intent, while also surfacing the features most associated with conversion.

## Dataset

[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) — 12,330 sessions with 18 features covering page views, durations, bounce/exit rates, and visitor metadata. Target: `Revenue` (purchase / no purchase).

## Approach

1. **EDA** — understand class imbalance, session patterns, and feature correlations
2. **Feature Engineering** — encode categoricals, scale numerics, address class imbalance
3. **Modeling** — train and tune Random Forest, XGBoost, and CatBoost individually
4. **Ensembling** — combine all three via soft voting / stacking
5. **Evaluation** — ROC-AUC, precision/recall, F1, SHAP-based interpretability
6. **Serving** — FastAPI backend exposing a `/predict` endpoint
7. **Frontend** — simple UI to input session data and view predicted purchase probability + key drivers

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn, XGBoost, CatBoost, imbalanced-learn, SHAP |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React (or Streamlit) |
| Tooling | pandas, numpy, matplotlib/seaborn/plotly, Optuna |

## Project Structure


**## Structure**

**- notebooks/ — EDA and modeling notebooks**

**- src/ — reusable pipeline code**

**- backend/ — FastAPI serving app**

**- frontend/ — UI**


## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Team

- **[Your name]** — Data pipeline, EDA, model training (RF, XGBoost, CatBoost), ensembling, evaluation
- **[Partner's name]** — Backend API, frontend, integration, deployment
# online-purchase-prediction

