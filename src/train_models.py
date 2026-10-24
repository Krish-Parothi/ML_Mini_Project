"""
State-of-the-Art Machine Learning Model Training & Comparative Analytics Pipeline
Trains, evaluates, and benchmarks 4+ SOTA models for Phishing URL Detection:
1. LightGBM Classifier (Gradient Boosted Trees - Leaf-wise)
2. XGBoost Classifier (Gradient Boosted Trees - Depth-wise Regularized)
3. Random Forest Classifier (Bagging Ensemble)
4. Logistic Regression (L2-Regularized Linear Baseline)
5. Decision Tree (Interpretable Non-linear Baseline)

Computes:
- Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, Log Loss
- Inference Latency (ms/sample) and Model Artifact Footprint (MB)
- Generates publication-ready figures and structured JSON/CSV metrics tables.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, log_loss
)

DATA_DIR = "data"
MODELS_DIR = "models"
FIGURES_DIR = "figures"

# Consistent publication color palette
PALETTE = {
    "LightGBM": "#00d2ff",
    "XGBoost": "#ff5252",
    "Random Forest": "#2ed573",
    "Logistic Regression": "#ffa502",
    "Decision Tree": "#9c88ff"
}

def load_data():
    print("[*] Loading preprocessed matrices from disk...", flush=True)
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    
    X_train_scaled = np.load(os.path.join(DATA_DIR, "X_train_scaled.npy"))
    X_val_scaled = np.load(os.path.join(DATA_DIR, "X_val_scaled.npy"))
    X_test_scaled = np.load(os.path.join(DATA_DIR, "X_test_scaled.npy"))
    
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    with open(os.path.join(DATA_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
        
    print(f"[+] Loaded Train ({len(X_train)}), Val ({len(X_val)}), Test ({len(X_test)}) with {len(feature_names)} features.", flush=True)
    return (X_train, X_val, X_test,
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, feature_names)

def train_and_benchmark_models():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    (X_train, X_val, X_test,
     X_train_scaled, X_val_scaled, X_test_scaled,
     y_train, y_val, y_test, feature_names) = load_data()
    
    # Model Configurations
    models_dict = {
        "LightGBM": {
            "model": LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            "use_scaled": False,
            "type": "Boosting Ensemble"
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            ),
            "use_scaled": False,
            "type": "Boosting Ensemble"
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False,
            "type": "Bagging Ensemble"
        },
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42
            ),
            "use_scaled": True,
            "type": "Linear Baseline"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=10,
                random_state=42
            ),
            "use_scaled": False,
            "type": "Tree Baseline"
        }
    }
    
    results = {}
    curves_data = {}
    best_f1 = -1.0
    best_model_name = ""
    best_model_obj = None

    print("\n" + "="*70, flush=True)
    print("STARTING MODEL TRAINING & EVALUATION MATRIX", flush=True)
    print("="*70, flush=True)
    
    for name, config in models_dict.items():
        print(f"\n[*] Training {name} ({config['type']})...", flush=True)
        clf = config["model"]
        
        # Select scaled or raw features
        X_tr = X_train_scaled if config["use_scaled"] else X_train
        X_te = X_test_scaled if config["use_scaled"] else X_test
        
        # Measure Training Time
        t_start = time.perf_counter()
        clf.fit(X_tr, y_train)
        train_time = time.perf_counter() - t_start
        
        # Measure Inference Latency over Test Set
        t_infer_start = time.perf_counter()
        y_pred = clf.predict(X_te)
        infer_total_time = time.perf_counter() - t_infer_start
        latency_per_sample_ms = (infer_total_time / len(X_te)) * 1000.0
        
        # Probabilities for ROC & PR curves
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_prob = clf.decision_function(X_te)
        else:
            y_prob = y_pred

        # Performance Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        loss = log_loss(y_test, y_prob)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Model artifact size on disk
        temp_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(clf, temp_path)
        model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        print(f" -> Accuracy:    {acc*100:.2f}%", flush=True)
        print(f" -> Precision:   {prec*100:.2f}%", flush=True)
        print(f" -> Recall:      {rec*100:.2f}%", flush=True)
        print(f" -> Specificity: {specificity*100:.2f}%", flush=True)
        print(f" -> F1-Score:    {f1*100:.2f}%", flush=True)
        print(f" -> ROC-AUC:     {auc:.4f}", flush=True)
        print(f" -> Train Time:  {train_time:.2f}s | Latency: {latency_per_sample_ms:.4f} ms/sample", flush=True)
        print(f" -> Model Size:  {model_size_mb:.2f} MB", flush=True)
        
        # Save curve points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
        
        curves_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_precision.tolist(),
            "pr_recall": pr_recall.tolist(),
            "auc": float(auc)
        }
        
        results[name] = {
            "model_type": config["type"],
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "log_loss": round(float(loss), 4),
            "train_time_sec": round(float(train_time), 2),
            "latency_ms_per_sample": round(float(latency_per_sample_ms), 4),
            "model_size_mb": round(float(model_size_mb), 2),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        
        # Check best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf
            
    print("\n" + "="*70, flush=True)
    print(f"[+] BEST OVERALL MODEL: {best_model_name} (F1-Score: {best_f1*100:.2f}%)", flush=True)
    print("="*70, flush=True)
    
    # Save best model
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.joblib"))
    best_meta = {
        "best_model_name": best_model_name,
        "best_f1_score": round(float(best_f1), 4),
        "use_scaled": models_dict[best_model_name]["use_scaled"],
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(MODELS_DIR, "best_model_meta.json"), "w") as f:
        json.dump(best_meta, f, indent=2)
        
    with open(os.path.join(FIGURES_DIR, "models_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV comparison table
    df_metrics = pd.DataFrame.from_dict(results, orient="index")
    df_metrics_clean = df_metrics[[
        "model_type", "accuracy", "precision", "recall", "specificity", "f1_score",
        "roc_auc", "train_time_sec", "latency_ms_per_sample", "model_size_mb"
    ]]
    df_metrics_clean.to_csv(os.path.join(FIGURES_DIR, "metrics_comparison_table.csv"))
    print(f"\n[+] Saved metrics comparison CSV to {os.path.join(FIGURES_DIR, 'metrics_comparison_table.csv')}", flush=True)

    # Generate Publication Figures
    generate_figures(curves_data, results, models_dict, feature_names, X_test, y_test)
    
    return results

def generate_figures(curves_data, results, models_dict, feature_names, X_test, y_test):
    print("\n[*] Generating high-resolution publication charts...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 1. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
    for name, data in curves_data.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {data['auc']:.4f})", lw=2.2, color=color)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Receiver Operating Characteristic (ROC) Benchmark Curves', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curves_comparison.png"))
    plt.close(fig)
    print(" [+] Generated: roc_curves_comparison.png", flush=True)

    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    for name, data in curves_data.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(data["pr_recall"], data["pr_precision"], label=f"{name}", lw=2.2, color=color)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Precision-Recall Curves for Phishing Threat Detection', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower left", fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pr_curves_comparison.png"))
    plt.close(fig)
    print(" [+] Generated: pr_curves_comparison.png", flush=True)

    # 3. Confusion Matrix Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    top_4_models = ["LightGBM", "XGBoost", "Random Forest", "Logistic Regression"]
    
    for ax, name in zip(axes.ravel(), top_4_models):
        cm_data = results[name]["confusion_matrix"]
        cm_array = np.array([
            [cm_data["tn"], cm_data["fp"]],
            [cm_data["fn"], cm_data["tp"]]
        ])
        # Annotate with percentages
        cm_sum = cm_array.sum()
        annot = np.array([
            [f"{cm_array[0,0]:,}\n({cm_array[0,0]/cm_sum*100:.1f}%)", f"{cm_array[0,1]:,}\n({cm_array[0,1]/cm_sum*100:.1f}%)"],
            [f"{cm_array[1,0]:,}\n({cm_array[1,0]/cm_sum*100:.1f}%)", f"{cm_array[1,1]:,}\n({cm_array[1,1]/cm_sum*100:.1f}%)"]
        ])
        sns.heatmap(cm_array, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{name}\nAcc: {results[name]['accuracy']*100:.2f}% | F1: {results[name]['f1_score']*100:.2f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label (0: Legitimate, 1: Phishing)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Actual Label (0: Legitimate, 1: Phishing)', fontsize=10, fontweight='bold')
        ax.set_xticklabels(['Legit', 'Phish'])
        ax.set_yticklabels(['Legit', 'Phish'])
        
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"))
    plt.close(fig)
    print(" [+] Generated: confusion_matrices.png", flush=True)

    # 4. Grouped Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    x = np.arange(len(labels))
    width = 0.16
    multiplier = 0
    
    for name, res in results.items():
        offset = width * multiplier
        values = [res[m] * 100 for m in metrics_to_plot]
        color = PALETTE.get(name, "#888888")
        rects = ax.bar(x + offset, values, width, label=name, color=color, edgecolor='black', alpha=0.9)
        multiplier += 1

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Model Metric Performance Analytics (Test Set: 12,000 URLs)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim([85, 103])
    ax.legend(loc='lower right', ncol=3, frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_metrics_barchart.png"))
    plt.close(fig)
    print(" [+] Generated: model_metrics_barchart.png", flush=True)

    # 5. Top 20 Feature Importances (from LightGBM)
    lgbm_model = models_dict["LightGBM"]["model"]
    if hasattr(lgbm_model, "feature_importances_"):
        importances = lgbm_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        top_features = [feature_names[i] for i in indices]
        top_scores = importances[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_scores[::-1], color='#00d2ff', edgecolor='#008bb2', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1], fontsize=10, fontweight='bold')
        ax.set_xlabel('Split Importance Gain / Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Predictive Phishing Indicators (LightGBM)', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
        plt.close(fig)
        print(" [+] Generated: feature_importance.png", flush=True)

    # 6. Latency vs Accuracy Trade-off Plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for name, res in results.items():
        color = PALETTE.get(name, "#888888")
        ax.scatter(res["latency_ms_per_sample"], res["accuracy"] * 100, s=res["model_size_mb"] * 40 + 150,
                   color=color, alpha=0.85, edgecolors='black', linewidth=1.5, label=name)
        ax.annotate(f"{name}\n({res['accuracy']*100:.1f}%, {res['latency_ms_per_sample']:.3f}ms)",
                    (res["latency_ms_per_sample"], res["accuracy"] * 100),
                    textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Inference Latency per Sample (Milliseconds)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Pareto Frontier: Inference Latency vs. Classification Accuracy\n(Bubble size proportional to model file size)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([88, 101])
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "latency_vs_accuracy.png"))
    plt.close(fig)
    print(" [+] Generated: latency_vs_accuracy.png", flush=True)

if __name__ == "__main__":
    train_and_benchmark_models()
