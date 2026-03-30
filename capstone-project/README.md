# Multi-Channel Marketing Attribution Model

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red?style=for-the-badge)
  ![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-brightgreen?style=for-the-badge)

</div>

## 📑 Table of Contents
- [Executive Summary](#executive-summary)
- [System Architecture](#system-architecture)
- [The Analytics Command Center (Web UI)](#the-analytics-command-center-web-ui)
- [Project Structure](#project-structure)
- [Installation Setup](#installation-setup)
- [Execution Guide](#execution-guide)
- [Methodology & Explainability](#methodology--explainability)
- [Business Impact](#business-impact)

---

## 🚀 Executive Summary

This repository contains a production-ready, data-driven marketing attribution pipeline and a **Real-Time Web Analytics Dashboard**. It solves a classic enterprise marketing challenge: **allocating campaign credit fairly across multi-touch customer journeys**. 

By circumventing simplistic "Last-Click" attribution models, this project orchestrates a **hybrid analytical system**:
1. **Predictive Layer (Micro-Level)**: An **XGBoost Classifier** predicts binary conversion probabilities for individual user journeys in real-time.
2. **Interpretability Layer**: **SHAP (SHapley Additive exPlanations)** decomposes predicting trees to isolate raw importance vectors.
3. **Sequential Layer (Macro-Level)**: A **Markov Chain** system calculates transitional "Removal Effects" to quantify the overarching importance of early-stage awareness channels, optimizing entire corporate budgets.

---

## 🏗 System Architecture

The pipeline is split into logical environments optimized for scalability and end-user adoption:

1. **Ingestion & Modeling (`training_pipeline.py`)**: Normalizes chronologies, engineers features, and fits the XGBoost algorithms, exporting a production `.pkl` model.
2. **Markov Engine (`markov_attribution.py`)**: Computes structural pathway dependencies and saves strategic attribution metrics to `attribution_comparison.json`.
3. **The API & Web Engine (`app.py`)**: A high-performance asynchronous **FastAPI** server that loads the models into memory and serves a completely reactive front-end dashboard for executives to dynamically interface with the ML models.

---

## 🖥 The Analytics Command Center (Web UI)

The project includes a stunning, glassmorphism-styled dashboard served natively via the browser.

* **Module 1: Real-Time Conversion Simulator:** Drag interactive sliders (Touchpoints, Channels, Time) to simulate user journeys. The UI seamlessly pings the XGBoost API and updates the central probability gauge instantly without page reloads.
* **Module 2: Strategic Budget Allocator:** Enter your total departmental budget (e.g., `$500,000`), and the system dynamically reads the Markov Chain attribution pathways to definitively allocate exactly how much money should go to Email, Search Ads, Social Media, etc., eliminating human guesswork.
* **Module 3: Internal Documentation:** A fully integrated User Manual providing in-depth glossaries on classification metrics and business applications.

---

## 📁 Project Structure

```text
capstone-project/
├── data/                      # Data assets and feature stores
│   ├── raw_data.csv           # Raw ingestion data
│   └── processed_data.csv     # Transformed feature set used for ML
│   
├── notebooks/                 # Jupyter notebooks for EDA and experimentation
│   ├── eda.ipynb              # Exploratory Data Analysis & Viz
│   ├── feature_engineering.ipynb # Feature creation and preprocessing
│   └── modeling.ipynb         # Model research and hyperparameter tuning
│
├── presentation/              # Project presentation slides and materials
│   └── capstone_presentation.pptx # Final stakeholders presentation
│
├── src/                       # Production-grade source code
│   ├── app.py                 # FastAPI backend & web server
│   ├── markov_attribution.py  # Markov logic & SHAP explainers
│   ├── training_pipeline.py   # Primary automation and orchestration script
│   ├── templates/             # HTML Jinja Templates (Dashboard & Manual)
│   └── static/                # CSS Stylesheets and JS Application Scripts
│
├── model/                     # Serialized deployment artifacts
│   └── trained_model.pkl      # Pickled production XGBoost classifier
│
└── README.md                  # System documentation
```

---

## ⚙️ Installation Setup

We recommend utilizing standard virtual environments. Ensure Python `3.9+` is available.

### 1. Provision the Environment
```bash
git clone https://github.com/organization/capstone-project.git
cd capstone-project
python -m venv venv
```

### 2. Activate the Environment
* **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
* **macOS / Linux (Bash):** `source venv/bin/activate`

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install fastapi uvicorn jinja2 pandas numpy scikit-learn xgboost matplotlib shap
```

---

## 💻 Execution Guide

### 1. Training the Models (Offline Pipeline)
To generate the model and build the Markov attribution matrices from raw data:
```bash
python src/training_pipeline.py
python src/markov_attribution.py
```

### 2. Launching the Web Application
To boot up the Analytics Command Center:
```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
* **Dashboard URL:** `http://127.0.0.1:8000/`
* **API Documentation:** `http://127.0.0.1:8000/docs`
* **User Manual:** `http://127.0.0.1:8000/manual`

---

## 🔬 Methodology & Explainability

### The "Last Click" Dilemma
By default, most digital platforms assign 100% of conversion revenue to the very last URL clicked (often a Search Ad). This causes early-stage engagement (Display Ads or Social Media) to incorrectly appear financially useless.

### Dual Measurement Paradigm
Our solution combines:
1. **XGBoost (Predictive View)**: Deconstructs gradient trees to evaluate real-time configurations dynamically via our UI simulator.
2. **Markov (Sequential View)**: Recreates full user histories (`Start -> Display -> Social -> Search -> Convert`). When we mathematically "remove" one node from the chain, we simulate what happens to the global conversion probability. If it collapses drastically, that channel gets high attribution credit across your budget.

---

## 📊 Business Impact

Deploying this architecture yields tangible C-suite outcomes:
* **Reduce CAC (Customer Acquisition Cost)**: By defunding heavily overrated last-click channels.
* **Optimize Top-of-Funnel budgets**: Re-justify Display Ad spend by proving its necessity upstream in the transition matrices.
* **Removes Guesswork**: Algorithms distribute exact budget allocations rapidly and adaptively based on verifiable hard statistics.

<div align="center">
<i>Engineered internally for strategic marketing optimization.</i>
</div>
