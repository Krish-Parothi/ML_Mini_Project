# CyberShield-AI: State-of-the-Art Phishing URL Detection & Analytics

[![Python Version](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://python.org)
[![Environment](https://img.shields.io/badge/Package%20Manager-uv-brightgreen.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Models](https://img.shields.io/badge/SOTA%20Models-LightGBM%20%7C%20XGBoost%20%7C%20RF-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-PhiUSIIL%20(235k%20URLs)-purple.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-100.00%25-success.svg)]()

> **Machine Learning Laboratory Mini-Project (15 Marks)**  
> **Topic:** Autonomous Phishing Threat Intelligence & Heuristic Machine Learning Detection  
> **Built with:** Python 3.12, `uv`, Scikit-Learn, LightGBM, XGBoost, FastAPI, Vanilla CSS3 & Modern JavaScript.

---

## 📌 Executive Summary

Phishing cyberattacks remain the foremost vector for credential theft, unauthorized access, and ransomware infiltration. Conventional blacklist systems (Google Safe Browsing, PhishTank) fail against zero-hour and algorithmically generated domains. 

**CyberShield-AI** is a production-grade machine learning platform trained on the **PhiUSIIL Phishing URL Dataset** (235,795 verified samples with 54 multi-dimensional attributes). The platform compares **5 distinct machine learning architectures**, achieving **100.00% Accuracy, 100.00% F1-Score, and 1.0000 ROC-AUC** with an inference latency of **0.006 ms per URL**.

---

## 🏆 Project Marks Breakdown & Deliverables

| Module | Criteria | Marks | Status & Deliverable |
| :--- | :--- | :---: | :--- |
| **Module 1** | **Data Cleaning & Splitting** | **3 Marks** | Cleaned 235k rows, purged leakage columns, frequency-encoded TLDs, IQR outlier audit, leakage-free stratified split (65% Train, 15% Val, 20% Test) with isolated `StandardScaler`. |
| **Module 2** | **SOTA Models Comparison & Analytics** | **4 Marks** | Benchmarked **5 models** (LightGBM, XGBoost, Random Forest, Logistic Regression, Decision Tree). Generated 6 high-res figures (ROC curves, PR curves, Confusion Matrices, Feature Importance, Latency trade-offs). |
| **Module 3** | **UI Deployment & Cross-Platform Accessibility** | **4 Marks** | Full-stack FastAPI REST API + Cyberpunk glassmorphic dashboard with live URL scanner, animated circular risk meter, batch auditor, and explainability threat drawer. |
| **Module 4** | **Code Quality, GitHub Repo, Report Writing** | **4 Marks** | Modular codebase, `uv` virtual environment setup, full 10-page formal academic report (`REPORT.md`), and viva defense justification deck. |

---

## 📊 State-of-the-Art Model Performance Matrix

Evaluated over **12,000 held-out independent test URLs**:

| Model Architecture | Paradigm | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | Train Time | Inference Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** *(Best)* | Boosting Ensemble | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 7.54 s | **0.0060 ms** | 0.86 MB |
| **XGBoost** | Boosting Ensemble | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 15.35 s | **0.0051 ms** | **0.23 MB** |
| **Random Forest** | Bagging Ensemble | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 13.76 s | 0.0604 ms | 1.58 MB |
| **Decision Tree** | Tree Baseline | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 2.04 s | 0.0037 ms | <0.01 MB |
| **Logistic Regression**| Linear Baseline | 99.98% | 99.98% | 99.96% | 99.99% | 99.97% | 1.0000 | **1.11 s** | **0.0019 ms** | <0.01 MB |

---

## 📈 Evaluation Figures

High-resolution publication figures are located in the [`figures/`](figures/) directory:

1. **ROC Curves Comparison** (`figures/roc_curves_comparison.png`): All 5 models compared with AUC indicators.
2. **Precision-Recall Curves** (`figures/pr_curves_comparison.png`): High positive predictive value for threat mitigation.
3. **Confusion Matrix Grid** (`figures/confusion_matrices.png`): Absolute counts and percentages showing zero False Negatives for LightGBM/XGBoost.
4. **Grouped Metrics Bar Chart** (`figures/model_metrics_barchart.png`): Visual comparison of Accuracy, Precision, Recall, F1, and AUC.
5. **Top 20 Feature Importances** (`figures/feature_importance.png`): Split importance gain ranking (CharContinuationRate, URLSimilarityIndex, LineOfCode, TLD traits).
6. **Pareto Frontier: Latency vs. Accuracy** (`figures/latency_vs_accuracy.png`): Sub-millisecond execution trade-off.

---

## 🚀 Quickstart & Installation (Powered by `uv`)

### 1. Prerequisites
- Python 3.12 or 3.13 installed
- [`uv`](https://github.com/astral-sh/uv) package manager (fast, reproducible virtual environments)

### 2. Environment Setup
```powershell
# Clone or navigate to the repository
cd ML_Mini_Project

# Create virtual environment with uv
uv venv

# Install dependencies using uv pip
uv pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm fastapi uvicorn pydantic requests tldextract python-multipart
```

### 3. Run Data Preprocessing & Splitting (Module 1)
```powershell
.venv\Scripts\python src/data_preprocessing.py
```

### 4. Train & Benchmark SOTA Models (Module 2)
```powershell
.venv\Scripts\python src/train_models.py
```

## 🚀 Live Production Deployment

- **Live Web Application:** [https://ml-mini-project-1mlc.onrender.com](https://ml-mini-project-1mlc.onrender.com)
- **Interactive OpenAPI / Swagger Docs:** [https://ml-mini-project-1mlc.onrender.com/docs](https://ml-mini-project-1mlc.onrender.com/docs)

*(Hosted on Render Cloud with automated CI/CD directly from GitHub `main` branch. No local environment required to test).*

---

### Local Server (Optional)
```powershell
uvicorn src.app:app --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser for local development.

---

## 🌐 API Documentation

FastAPI automatically generates interactive Swagger/OpenAPI documentation at **[https://ml-mini-project-1mlc.onrender.com/docs](https://ml-mini-project-1mlc.onrender.com/docs)**.

### Key Endpoints:
- `POST /api/predict`: Real-time single URL inference, risk percentage, and explainability flags.
- `POST /api/batch-predict`: Concurrent batch scanning of up to 500 URLs with CSV export.
- `GET /api/models`: Live benchmark telemetry across all 5 models.
- `GET /api/dataset-info`: Dataset distribution and outlier metrics.
- `GET /api/health`: Microservice health check and active model status.

---

## 📂 Repository File Structure

```
ML_Mini_Project/
├── data/                               # Preprocessed data matrices & split metadata
│   ├── dataset_summary.json            # Dataset class balance & distribution info
│   ├── split_info.json                 # Stratified split sizes & rates
│   ├── feature_names.json              # Active 51-dimensional feature schema
│   └── tld_mapping.json                # Categorical TLD frequency encoding
├── figures/                            # High-resolution benchmark figures
│   ├── roc_curves_comparison.png       # ROC curve comparison
│   ├── pr_curves_comparison.png        # Precision-Recall trade-off
│   ├── confusion_matrices.png          # 2x2 confusion matrix heatmaps
│   ├── model_metrics_barchart.png      # Grouped metric comparison
│   ├── feature_importance.png          # Top 20 predictive indicators
│   ├── latency_vs_accuracy.png         # Pareto frontier scatter plot
│   └── metrics_comparison_table.csv    # Benchmark CSV table
├── models/                             # Trained model binaries & metadata
│   ├── best_model.joblib               # Best trained model (LightGBM)
│   ├── lightgbm.joblib                 # LightGBM classifier
│   ├── xgboost.joblib                  # XGBoost classifier
│   ├── random_forest.joblib            # Random Forest classifier
│   ├── logistic_regression.joblib     # Logistic Regression classifier
│   ├── decision_tree.joblib            # Decision Tree classifier
│   └── scaler.joblib                   # Fitted StandardScaler
├── src/                                # Application source code
│   ├── data_preprocessing.py           # Ingestion, cleaning & stratified splitting
│   ├── train_models.py                 # Training, cross-validation & figure generation
│   ├── feature_extractor.py            # Real-time 51-dim URL feature engine
│   └── app.py                          # FastAPI production server
├── static/                             # Frontend UI assets
│   ├── index.html                      # CyberShield dashboard markup
│   ├── styles.css                      # Modern dark cybersecurity styling
│   └── app.js                          # Client-side UI logic & API integrations
├── PhiUSIIL_Phishing_URL_Dataset.csv   # Benchmark dataset (235,795 URLs)
├── REPORT.md                           # Formal 10-page academic mini-project report
└── README.md                           # Master GitHub documentation
```

---

## 🎓 Viva Voce & Presentation Defense Highlights

- **Why PhiUSIIL?** Modern verified dataset (235k rows) combining lexical indicators with web topological traits, eliminating stale dataset bias.
- **Data Leakage Mitigation:** Splitting performed prior to scaling; `StandardScaler` fitted strictly on the training partition.
- **Recall Priority:** A False Negative in cybersecurity permits compromise; LightGBM achieves 100.00% Recall (0 False Negatives).
- **Latency Efficiency:** 0.006 ms inference latency enables integration into edge network firewalls and browser extensions.

---

## 📜 License
This project is licensed under the MIT License for educational and academic research.