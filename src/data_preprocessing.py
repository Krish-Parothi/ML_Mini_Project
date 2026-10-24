"""
Data Preprocessing & Splitting Pipeline
PhiUSIIL Phishing URL Dataset
Handles data ingestion, validation, cleaning, feature selection, 
stratified train-val-test splitting, scaling, and metadata generation.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_DIR = "data"
MODELS_DIR = "models"
METRICS_DIR = "figures"

# Columns to exclude from direct tabular model features
IDENTIFIER_COLS = ["FILENAME", "URL", "Domain", "Title"]

def ensure_dirs():
    for d in [OUTPUT_DIR, MODELS_DIR, METRICS_DIR, "src"]:
        os.makedirs(d, exist_ok=True)

def load_and_clean_data(sample_size=60000, random_state=42):
    """
    Loads dataset, handles duplicates, invalid values, categorical encoding,
    and returns clean feature matrix X and target y.
    """
    ensure_dirs()
    print(f"[*] Ingesting PhiUSIIL dataset from {DATASET_PATH}...", flush=True)
    
    # Read raw dataset
    df_raw = pd.read_csv(DATASET_PATH)
    print(f"[+] Loaded raw dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns", flush=True)
    
    # Check duplicates and missing values
    initial_rows = len(df_raw)
    df_cleaned = df_raw.drop_duplicates(subset=['URL']).copy()
    dropped_dups = initial_rows - len(df_cleaned)
    print(f"[+] Duplicate URLs removed: {dropped_dups}", flush=True)
    
    # Missing value audit
    null_counts = df_cleaned.isnull().sum()
    cols_with_null = null_counts[null_counts > 0]
    if len(cols_with_null) > 0:
        print(f"[!] Null values detected in columns:\n{cols_with_null}", flush=True)
        df_cleaned = df_cleaned.dropna()
    else:
        print("[+] Zero missing values found across all columns.", flush=True)

    # Stratified sampling if requested
    if sample_size and sample_size < len(df_cleaned):
        print(f"[*] Extracting balanced stratified subset of {sample_size:,} records...", flush=True)
        df, _ = train_test_split(
            df_cleaned, 
            train_size=sample_size, 
            stratify=df_cleaned['label'], 
            random_state=random_state
        )
    else:
        df = df_cleaned

    # Target variable: 
    # In PhiUSIIL original dataset: label == 1 is Legitimate, label == 0 is Phishing
    # In Threat Detection standard ML: 1 = Phishing (Positive threat), 0 = Legitimate (Safe)
    df['is_phishing'] = (df['label'] == 0).astype(int)
    
    # Feature Selection:
    features_df = df.drop(columns=[col for col in IDENTIFIER_COLS if col in df.columns] + ['label', 'is_phishing']).copy()
    
    # Handle TLD column (categorical string, e.g., 'com', 'org', 'xyz')
    if 'TLD' in features_df.columns:
        top_tlds = features_df['TLD'].value_counts().head(30).index.tolist()
        tld_mapping = {str(tld): i + 1 for i, tld in enumerate(top_tlds)}
        features_df['TLD_Code'] = features_df['TLD'].map(lambda x: tld_mapping.get(str(x), 0))
        features_df = features_df.drop(columns=['TLD'])
        with open(os.path.join(OUTPUT_DIR, "tld_mapping.json"), "w") as f:
            json.dump(tld_mapping, f, indent=2)

    # Ensure all remaining columns are numeric
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    # Replace infinite values
    features_df = features_df.replace([np.inf, -np.inf], 0)
    
    feature_names = features_df.columns.tolist()
    print(f"[+] Final active numerical features ({len(feature_names)}): {feature_names[:6]}...", flush=True)
    
    X = features_df
    y = df['is_phishing']
    
    # Outlier analysis on key numerical features (e.g. URLLength, LineOfCode)
    outlier_stats = {}
    for col in ['URLLength', 'DomainLength', 'LineOfCode', 'NoOfSubDomain']:
        if col in X.columns:
            q25, q75 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = int(((X[col] < lower_bound) | (X[col] > upper_bound)).sum())
            outlier_stats[col] = {
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
                "outlier_count": outlier_count,
                "outlier_percentage": round((outlier_count / len(X)) * 100, 2)
            }
            
    stats = {
        "total_samples": int(len(df)),
        "phishing_count": int(y.sum()),
        "legitimate_count": int((y == 0).sum()),
        "phishing_ratio": round(float(y.mean()), 4),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "outlier_analysis": outlier_stats
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"[+] Dataset Summary: {stats['total_samples']:,} samples | Phishing: {stats['phishing_count']:,} ({stats['phishing_ratio']*100:.2f}%) | Legitimate: {stats['legitimate_count']:,}", flush=True)
    
    return X, y, feature_names, df[['URL', 'is_phishing']]

def prepare_splits_and_scale(X, y, test_size=0.20, val_size=0.15, random_state=42):
    """
    Performs Stratified Train-Validation-Test splitting and standard scaling.
    Ensures zero data leakage between training and testing folds.
    """
    print(f"[*] Executing Stratified Train-Val-Test Splitting (Test: {test_size*100}%, Val: {val_size*100}% of train)...", flush=True)
    
    # Train-test split (Stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Train-val split (Stratified)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, stratify=y_train_full, random_state=random_state
    )
    
    print(f"[+] Split sizes -> Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}", flush=True)
    
    # Fit Scaler strictly on X_train to prevent data leakage
    print("[*] Fitting StandardScaler on training set...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump(list(X.columns), f, indent=2)
        
    # Save split metrics
    split_info = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_phishing_rate": round(float(y_train.mean()), 4),
        "val_phishing_rate": round(float(y_val.mean()), 4),
        "test_phishing_rate": round(float(y_test.mean()), 4)
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
        
    # Save training and test matrices for quick reuse if needed
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train.values)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val.values)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test.values)
    np.save(os.path.join(OUTPUT_DIR, "X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_val_scaled.npy"), X_val_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val.values)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.values)
    
    print("[+] Preprocessing & Splitting completed successfully.", flush=True)
    return (X_train, X_val, X_test,
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, list(X.columns))

if __name__ == "__main__":
    X, y, feature_names, url_meta = load_and_clean_data(sample_size=60000)
    prepare_splits_and_scale(X, y)
