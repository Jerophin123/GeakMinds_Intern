# Multi-Channel Marketing Attribution Model

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project provides a robust, data-driven marketing attribution model designed to analyze and quantify the impact of user touchpoints across diverse marketing channels (Email, Social Media, Search Ads, Display Ads). By estimating each channel's true contribution to user conversions, this solution empowers marketing teams to optimize their budget allocation and drive higher return on ad spend (ROAS).

---

## Architecture and Project Structure

### Repository Layout
```text
capstone-project/
├── data/                      # Data assets (Not tracked in version control)
│   ├── raw_data.csv           # Raw ingestion data
│   └── processed_data.csv     # Transformed feature set used for modeling
├── notebooks/                 # Exploratory and experimental notebooks
│   ├── eda.ipynb              # Exploratory Data Analysis & visual insights
│   ├── feature_engineering.ipynb  # Feature extraction workflows
│   └── modeling.ipynb         # Model training, hyperparameter tuning & SHAP
├── src/                       # Production-grade source code
│   └── training_pipeline.py   # Automated end-to-end model training script
├── model/                     # Model artifacts natively serialized
│   └── trained_model.pkl      # Production-ready trained XGBoost classifier
└── README.md                  # Project documentation
```

### Dataset Semantics
The analytical dataset comprises sequential user touchpoints capturing the customer journey:
- `User_ID`: Unique customer identifier.
- `Timestamp`: Datetime of the interaction.
- `Channel`: The marketing medium (Email, Social Media, Search Ads, Display Ads).
- `Campaign`: The granular campaign identifier.
- `Conversion`: Binary target variable (1 = Converted, 0 = Did not convert).

---

## Quickstart & Installation

Follow these instructions to set up the environment locally.

### 1. Environment Setup

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**macOS/Linux (Bash):**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
Install the required data science and machine learning packages:
```bash
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap jupyter
```

---

## Execution Guide

### Option 1: Automated ML Pipeline (Recommended for Production)
Execute the end-to-end training pipeline via the command line. This script automatically performs data loading, feature engineering, model training (XGBoost), evaluation, and serialization.

```bash
# Run from the project root directory
python src/training_pipeline.py
```
**Expected Output:**
- Generates `data/processed_data.csv`
- Serializes the trained classifier to `model/trained_model.pkl`
- Prints evaluation metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) to standard output.

### Option 2: Interactive Analysis (Jupyter Notebooks)
For exploratory data analysis, visual insights, and interactive hyperparameter tuning:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and execute the notebooks in the following logical sequence:
1. `eda.ipynb`
2. `feature_engineering.ipynb`
3. `modeling.ipynb`

---

## Model Evaluation Metrics
The modeling pipeline standardizes on the **XGBoost Classifier**, selected for its high performance on tabular data with non-linear relationships.

The pipeline automatically calculates and reports the following key metrics on a holdout test set:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Accuracy of positive predictions (minimizing false positives).
- **Recall**: Ability to find all positive instances (minimizing false negatives).
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Aggregate measure of performance across all classification thresholds.

---

## Strategic Business Insights & SHAP Explainability

Leveraging SHAP (SHapley Additive exPlanations), the model provides interpretable insights into feature importance and channel efficacy.

### Attribution Discoveries
*   **Search Ads**: Exhibits high-intent characteristics, frequently serving as the decisive final touchpoint preceding conversion.
*   **Email**: A highly effective retargeting channel, demonstrating a strong conversion rate among returning/engaged users.
*   **Social Media**: Functions exceptionally well for mid-funnel engagement and brand reinforcement.
*   **Display Ads**: Operates primarily as an upper-funnel awareness driver; most impactful as an introductory touchpoint.

### Strategic Recommendations
1.  **Fund High-Intent Channels**: Maximize budget allocation to Search Ads to effectively capture bottom-of-funnel users demonstrating active purchase intent.
2.  **Optimize Email Funnels**: Capitalize on Email's strong closing probability by optimizing automated welcome series and cart-abandonment retargeting sequences.
3.  **Reframe Display KPIs**: Transition away from evaluating Display Ads strictly on direct, last-click conversions. Measure Display Ad success based on top-of-funnel entry metrics and downstream retargetable audience generation.
