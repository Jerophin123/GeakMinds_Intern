# Multi-Channel Marketing Attribution Model

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red?style=for-the-badge)
  ![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-brightgreen?style=for-the-badge)

</div>

## 📑 Table of Contents
- [Executive Summary](#executive-summary)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Dataset Specifications](#dataset-specifications)
- [Installation Setup](#installation-setup)
- [Execution Guide](#execution-guide)
- [Methodology & Explainability](#methodology--explainability)
- [Business Impact](#business-impact)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Contributing](#contributing)

---

## 🚀 Executive Summary

This repository contains a production-ready, data-driven marketing attribution pipeline. It solves a classic enterprise marketing challenge: **allocating campaign credit fairly across multi-touch customer journeys**. 

By circumventing simplistic "Last-Click" attribution models, this project orchestrates a **hybrid analytical system**:
1. **Predictive Layer**: An **XGBoost Classifier** predicts binary conversion probabilities based on tabular feature engineering.
2. **Interpretability Layer**: **SHAP (SHapley Additive exPlanations)** decomposes predicting trees to isolate raw importance vectors.
3. **Sequential Layer**: A **Markov Chain Multi-Touch Attribution (MTA)** system calculates transitional "Removal Effects" to quantify the overarching importance of early-stage awareness channels (like Display Ads) that fall prey to end-touch biases.

---

## 🏗 System Architecture

The pipeline is split into distinct logical environments optimized for scalability:

1. **Ingestion & Processing (`transform_features`)**: Normalizes timestamps, aggregates unique user funnels, extracts path lengths, and one-hot encodes categorical channels.
2. **Modeling Engine (`train_model`)**: Fits configured XGBoost algorithms to training splits, computing global accuracy, Precision, Recall, F1, and cross-threshold ROC-AUC metrics.
3. **Interpretability Pipeline (`markov_attribution.py`)**: Simultaneously models chronological matrices and interprets trained SHAP artifacts to surface real-time actionable recommendations.

---

## 📁 Project Structure

The codebase is logically partitioned separating raw datasets, exploration notebooks, and automated source code.

```text
capstone-project/
├── data/                      # Data assets and feature stores
│   ├── raw_data.csv           # Raw ingestion data
│   └── processed_data.csv     # Transformed feature set used for ML
│   
├── notebooks/                 # Exploratory and experimental sandboxes
│   ├── eda.ipynb              # Visual insights & anomaly detection
│   ├── feature_engineering.ipynb  # Feature extraction workflows
│   └── modeling.ipynb         # Hyperparameter tuning (GridSearchCV)
│
├── src/                       # Production-grade source code
│   ├── markov_attribution.py  # Markov logic, SHAP explainers & plotting
│   └── training_pipeline.py   # Primary automation and orchestration script
│
├── model/                     # Serialized deployment artifacts
│   └── trained_model.pkl      # Pickled production XGBoost classifier
│
├── ppt_utf8.txt               # Internal documentation / Transcripts
└── README.md                  # System documentation
```

---

## 🗄 Dataset Specifications

The expected input to `data/raw_data.csv` must follow this schema for the pipeline to compile smoothly without type errors.

| Feature Name | Primary Data Type | Description | Example Target |
| :--- | :--- | :--- | :--- |
| `User_ID` | `Integer` | Universally Unique Identifier for an engaged lead. | `83281` |
| `Timestamp` | `Datetime` | The ISO-8601 absolute timestamp of the touch. | `2025-02-10 07:58:51` |
| `Channel` | `String` | Categorical digital property interacted with. | `Search Ads`, `Email` |
| `Campaign` | `String` | Unique naming convention for the marketing push. | `Winter Sale 2025` |
| `Conversion` | `String/Int` | Binary classification mapping representing success. | `Yes` / `1` |

*(The `src/training_pipeline.py` script automatically maps alphanumeric conversion mappings to machine-readable integers.)*

---

## ⚙️ Installation Setup

We recommend utilizing standard virtual environments. Ensure Python `3.9+` is available via your system `$PATH`.

### 1. Provision the Environment
```bash
# Clone the enterprise repository
git clone https://github.com/organization/capstone-project.git
cd capstone-project

# Initialize a clean python virtual environment
python -m venv venv
```

### 2. Activate the Environment
* **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
* **macOS / Linux (Bash):** `source venv/bin/activate`

### 3. Install Dependencies
```bash
# Upgrade foundational packages
python -m pip install --upgrade pip

# Install required numerical, modeling, and plotting libraries
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
```

---

## 💻 Execution Guide

### Unattended Production Execution
The entire pipeline can be orchestrated via a single terminal command. This is suitable for Cron jobs, Airflow DAGs, or generalized CI/CD workflow triggers.

```bash
python src/training_pipeline.py
```
**Triggering the script natively performs:**
1. Real-time chronological sorting and feature engineering.
2. In-memory data splits (`test_size=0.2`).
3. Supervised classification training.
4. Exporting native JSON diagnostics, compiled model pkl formats, and Matplotlib comparative PNG graphs to the `/data/` and `/model/` directories.

### Interactive Sandboxing
Data scientists wishing to tune specific XGBoost learning rates or `max_depth` arrays interactively should boot Jupyter inside the root node:
```bash
jupyter notebook
```
Follow the logical progression: `eda.ipynb` -> `feature_engineering.ipynb` -> `modeling.ipynb`.

---

## 🔬 Methodology & Explainability

### The "Last Click" Dilemma
By default, most digital platforms assign 100% of conversion revenue to the very last URL clicked (often a generic Google Search Ad or an Email). This causes early-stage engagement (Display Ads or Social Media) to incorrectly appear financially useless.

### Dual Measurement Paradigm
Our solution combines:
1. **SHAP (Predictive View)**: Deconstructs the gradient trees to see which mathematically engineered features (e.g., `time_to_conversion_hours` vs `first_channel`) hold maximum predictive weight. 
2. **Markov (Sequential View)**: Recreates full user histories (`Start -> Display -> Social -> Search -> Convert`). When we mathematically "remove" one node from the chain (Removal Effect Analysis), we simulate what happens to the global conversion probability. If it collapses drastically, that channel gets high attribution credit.

---

## 📊 Business Impact

Deploying this architecture yields tangible C-suite outcomes:
* **Reduce CAC (Customer Acquisition Cost)**: By defunding heavily overrated last-click channels.
* **Optimize Top-of-Funnel budgets**: Re-justify Display Ad spend by proving its necessity upstream in the transition matrices.
* **Granular Precision**: 85%+ validation accuracy guarantees programmatic adjustments are financially safe.

---

## 🛡 Testing & Quality Assurance

If deploying this alongside automated tests (e.g. `pytest`), it is best practice to validate the integrity of outputs prior to model promotion:
* Ensure output dimensions of `data/processed_data.csv` match expected grouping behaviors `len(df_raw['User_ID'].unique()) == len(df_processed)`.
* Enforce `ROC_AUC >= 0.85` strictly during threshold monitoring prior to `.pkl` overwrites.

---

## 🤝 Contributing

We welcome internal MRs/PRs to enhance the capabilities of the attribution engine.
1. Branch off `main` as `feature/your-feature-name`.
2. Ensure new Python scripts adhere to [PEP 8](https://peps.python.org/pep-0008/) style standards.
3. If new packages are utilized, update the installation blocks inside this `README.md`.
4. Open a Merge Request assigned to the Data Science governing body.

<div align="center">
<i>Engineered internally by Jerophin D R - Data Science Intern </i>
</div>
