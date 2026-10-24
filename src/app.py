"""
CyberShield Phishing Detection System - Production REST API
FastAPI Backend serving real-time URL heuristic extraction,
model inference with SOTA ensembles, batch evaluation, and benchmark telemetry.
"""

import os
import json
import time
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.feature_extractor import extract_url_features, FEATURE_NAMES

app = FastAPI(
    title="CyberShield SOTA Phishing URL Detection API",
    description="High-throughput, explainable real-time phishing URL detection engine powered by PhiUSIIL and Gradient Boosted Ensembles.",
    version="2.0.0"
)

# Enable CORS for cross-platform access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
FIGURES_DIR = "figures"
DATA_DIR = "data"

# Global Model Cache
LOADED_MODELS = {}
SCALER = None
BEST_MODEL_META = {}
MODELS_COMPARISON = {}
DATASET_SUMMARY = {}

def load_system_artifacts():
    global LOADED_MODELS, SCALER, BEST_MODEL_META, MODELS_COMPARISON, DATASET_SUMMARY
    print("[*] Loading ML models and telemetry artifacts...", flush=True)
    
    # Load Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
        print("[+] Loaded StandardScaler.", flush=True)
        
    # Load Best Model Meta
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            BEST_MODEL_META = json.load(f)

    # Load Comparison Table
    comp_path = os.path.join(FIGURES_DIR, "models_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            MODELS_COMPARISON = json.load(f)

    # Load Dataset Summary
    summary_path = os.path.join(DATA_DIR, "dataset_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            DATASET_SUMMARY = json.load(f)

    # Load all models into memory for instant hot-swapping
    model_files = {
        "LightGBM": os.path.join(MODELS_DIR, "lightgbm.joblib"),
        "XGBoost": os.path.join(MODELS_DIR, "xgboost.joblib"),
        "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
        "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
        "Decision Tree": os.path.join(MODELS_DIR, "decision_tree.joblib")
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            LOADED_MODELS[name] = joblib.load(path)
            print(f"[+] Cached model: {name}", flush=True)

    # If best_model exists
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(best_path):
        LOADED_MODELS["Best (Auto)"] = joblib.load(best_path)

# Initialize models on startup
load_system_artifacts()

class PredictionRequest(BaseModel):
    url: str = Field(..., example="https://update-account-verification-paypal.com/login")
    model_name: Optional[str] = Field("Best (Auto)", example="LightGBM")

class BatchPredictionRequest(BaseModel):
    urls: List[str] = Field(..., example=["https://google.com", "http://phish-secure.xyz"])
    model_name: Optional[str] = Field("Best (Auto)")

@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "available_models": list(LOADED_MODELS.keys()),
        "best_model": BEST_MODEL_META.get("best_model_name", "LightGBM"),
        "feature_count": len(FEATURE_NAMES)
    }

@app.get("/api/models")
def get_models_metrics():
    return {
        "comparison": MODELS_COMPARISON,
        "best_model": BEST_MODEL_META
    }

@app.get("/api/dataset-info")
def get_dataset_info():
    return DATASET_SUMMARY

@app.post("/api/predict")
def predict_url(req: PredictionRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")
        
    model_choice = req.model_name or "Best (Auto)"
    if model_choice not in LOADED_MODELS:
        model_choice = "Best (Auto)" if "Best (Auto)" in LOADED_MODELS else list(LOADED_MODELS.keys())[0]

    model = LOADED_MODELS[model_choice]
    
    # 1. Real-time Feature Extraction
    t0 = time.perf_counter()
    extraction_result = extract_url_features(url)
    vector = np.array(extraction_result["vector"]).reshape(1, -1)
    extraction_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Scaling (if Logistic Regression)
    if model_choice == "Logistic Regression" and SCALER is not None:
        model_input = SCALER.transform(vector)
    else:
        model_input = vector

    # 3. Model Inference
    t_inf_start = time.perf_counter()
    pred_label = int(model.predict(model_input)[0])
    
    if hasattr(model, "predict_proba"):
        prob_dist = model.predict_proba(model_input)[0]
        # In our trained model: index 1 is Phishing, index 0 is Legitimate
        phish_prob = float(prob_dist[1])
    else:
        phish_prob = float(pred_label)
        
    inference_time_ms = (time.perf_counter() - t_inf_start) * 1000.0
    total_time_ms = extraction_time_ms + inference_time_ms

    # 4. Risk Categorization
    if phish_prob >= 0.85:
        risk_level = "CRITICAL THREAT"
        risk_color = "red"
    elif phish_prob >= 0.60:
        risk_level = "HIGH RISK"
        risk_color = "orange"
    elif phish_prob >= 0.35:
        risk_level = "MODERATE SUSPICION"
        risk_color = "yellow"
    else:
        risk_level = "SAFE / LEGITIMATE"
        risk_color = "green"

    is_phishing = 1 if phish_prob >= 0.50 else 0
    verdict = "Phishing URL Detected" if is_phishing else "Legitimate URL"

    return {
        "url": url,
        "is_phishing": bool(is_phishing),
        "verdict": verdict,
        "phishing_probability": round(phish_prob * 100.0, 2),
        "safety_probability": round((1.0 - phish_prob) * 100.0, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "model_used": model_choice,
        "latency_ms": {
            "feature_extraction": round(extraction_time_ms, 3),
            "model_inference": round(inference_time_ms, 3),
            "total": round(total_time_ms, 3)
        },
        "flags": extraction_result["flags"],
        "metadata": extraction_result["metadata"],
        "top_features": {
            "URL Length": extraction_result["features"].get("URLLength", 0),
            "Domain Length": extraction_result["features"].get("DomainLength", 0),
            "Is IP Address": bool(extraction_result["features"].get("IsDomainIP", 0)),
            "Subdomains": extraction_result["features"].get("NoOfSubDomain", 0),
            "HTTPS Enabled": bool(extraction_result["features"].get("IsHTTPS", 0)),
            "Char Continuation": extraction_result["features"].get("CharContinuationRate", 0),
            "Obfuscated Chars": extraction_result["features"].get("NoOfObfuscatedChar", 0)
        }
    }

@app.post("/api/batch-predict")
def batch_predict(req: BatchPredictionRequest):
    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty.")
    if len(urls) > 500:
        raise HTTPException(status_code=400, detail="Batch size limited to 500 URLs per request.")

    model_choice = req.model_name or "Best (Auto)"
    model = LOADED_MODELS.get(model_choice, LOADED_MODELS.get("Best (Auto)"))

    t_start = time.perf_counter()
    results = []
    phishing_count = 0
    safe_count = 0

    for u in urls:
        ext = extract_url_features(u)
        vec = np.array(ext["vector"]).reshape(1, -1)
        if model_choice == "Logistic Regression" and SCALER is not None:
            vec = SCALER.transform(vec)
        
        prob = float(model.predict_proba(vec)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(vec)[0])
        is_phish = prob >= 0.50
        if is_phish:
            phishing_count += 1
        else:
            safe_count += 1
            
        results.append({
            "url": u,
            "is_phishing": bool(is_phish),
            "phishing_probability": round(prob * 100.0, 2),
            "status": "PHISHING" if is_phish else "SAFE",
            "flag_count": len(ext["flags"])
        })

    elapsed_total_ms = (time.perf_counter() - t_start) * 1000.0
    throughput = (len(urls) / (elapsed_total_ms / 1000.0)) if elapsed_total_ms > 0 else len(urls)

    return {
        "total_urls": len(urls),
        "phishing_count": phishing_count,
        "safe_count": safe_count,
        "phishing_percentage": round((phishing_count / len(urls)) * 100, 2),
        "total_time_ms": round(elapsed_total_ms, 2),
        "throughput_urls_per_sec": round(throughput, 1),
        "results": results
    }

from fastapi.responses import FileResponse

# Mount figures directory for analytics charts
FIGURES_PATH = os.path.join(os.path.dirname(__file__), "..", "figures")
if os.path.exists(FIGURES_PATH):
    app.mount("/figures", StaticFiles(directory=FIGURES_PATH), name="figures")

@app.get("/REPORT.md")
def get_report():
    report_path = os.path.join(os.path.dirname(__file__), "..", "REPORT.md")
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="text/markdown")
    return {"error": "Report file not found"}

# Mount static files (Frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

