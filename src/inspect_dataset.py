import pandas as pd
import numpy as np

print("Loading PhiUSIIL dataset preview...")
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', nrows=20000)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

print("\nClass distribution (in 20,000 sample):")
print(df['label'].value_counts())

sample_legit = df[df['label'] == 1]['URL'].head(5).tolist()
sample_phish = df[df['label'] == 0]['URL'].head(5).tolist()

print("\nSample Label 1 URLs (Legitimate):")
for u in sample_legit:
    print(" -", u)

print("\nSample Label 0 URLs (Phishing):")
for u in sample_phish:
    print(" -", u)

print("\nChecking nulls:")
print(df.isnull().sum()[df.isnull().sum() > 0])
