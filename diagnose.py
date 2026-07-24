"""
Diagnostic script: checks whether the fake-news model is overfitting
to dataset-specific artifacts (e.g. Reuters-style datelines) instead
of learning genuine fake-vs-real content signals.

Run this from your project folder:
    python diagnose.py

Make sure fake.csv and true.csv are in the same folder this script
expects (edit DATA_DIR below if needed), and that models/*.pkl exist.
"""

import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MODELS_PATH = "models"
DATA_DIR = "data"  # CSV files are in data/ subfolder

# ---------------------------------------------------------------
# 1. Load models
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading models")
print("=" * 70)

with open(f"{MODELS_PATH}/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(f"{MODELS_PATH}/rf_baseline.pkl", "rb") as f:
    rf_baseline = pickle.load(f)

with open(f"{MODELS_PATH}/rf_pso.pkl", "rb") as f:
    rf_pso = pickle.load(f)

with open(f"{MODELS_PATH}/selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

print("Models loaded OK.")
print(f"Classes: {rf_baseline.classes_}  (confirm which number = fake / real "
      f"based on how you set fake_df['label'] / true_df['label'])")

# ---------------------------------------------------------------
# 2. Check accuracy on the model's own held-out test split
#    (this tells us if it's genuinely 99% accurate on data it
#    has never directly trained on, but drawn from the SAME dataset)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Accuracy on held-out split from the SAME dataset")
print("=" * 70)

try:
    fake_df = pd.read_csv(f"{DATA_DIR}/fake.csv")
    true_df = pd.read_csv(f"{DATA_DIR}/true.csv")

    fake_df["label"] = 0  # match whatever you set in train_models.py
    true_df["label"] = 1  # match whatever you set in train_models.py

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    text_col = "text" if "text" in df.columns else df.columns[0]

    X_all = vectorizer.transform(df[text_col].astype(str))
    y_all = df["label"].values

    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    preds = rf_baseline.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Baseline accuracy on held-out same-dataset split: {acc:.4f}")

    X_test_pso = X_test[:, selected_features]
    preds_pso = rf_pso.predict(X_test_pso)
    acc_pso = (preds_pso == y_test).mean()
    print(f"PSO accuracy on held-out same-dataset split: {acc_pso:.4f}")

except FileNotFoundError as e:
    print(f"Skipped (couldn't find fake.csv / true.csv): {e}")

# ---------------------------------------------------------------
# 3. Inspect top TF-IDF feature importances
#    If words like 'reuters', 'washington', dateline cities, etc.
#    dominate, the model is likely keying on formatting, not content.
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Top 30 most important features (baseline model)")
print("=" * 70)

feature_names = np.array(vectorizer.get_feature_names_out())
importances = rf_baseline.feature_importances_
top_idx = np.argsort(importances)[::-1][:30]

for rank, idx in enumerate(top_idx, 1):
    print(f"{rank:2d}. {feature_names[idx]:20s}  importance={importances[idx]:.5f}")

print("\n>>> Look for words like: reuters, washington, said, wire, city names, etc.")
print(">>> If these dominate, the model learned FORMAT, not CONTENT.")

# ---------------------------------------------------------------
# 4. Test on genuinely new/unseen text (hand-written below)
#    Add/replace with your own real & fake test sentences.
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Predictions on fresh, out-of-distribution text")
print("=" * 70)

test_samples = {
    "REAL (no Reuters formatting)": [
        "The city council voted Tuesday to approve a new budget for road repairs next year.",
        "Scientists at the university published new findings on renewable energy storage this week.",
        "The local school district announced changes to its bus routes ahead of the fall semester.",
    ],
    "FAKE (obviously fabricated)": [
        "Scientists confirm the moon is turning into cheese, NASA refuses to comment.",
        "Doctors hate this one trick: drinking bleach cures every disease, secret report claims.",
        "Government admits weather is controlled by a hidden machine in Area 51.",
    ],
}

for category, samples in test_samples.items():
    print(f"\n--- {category} ---")
    for text in samples:
        X = vectorizer.transform([text])
        pred = rf_baseline.predict(X)[0]
        proba = rf_baseline.predict_proba(X)[0]
        print(f"Text: {text[:60]}...")
        print(f"  Prediction: {pred}  Probabilities: {proba}")

print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print("""
- If STEP 2 accuracy is ~95-99% but STEP 4 predictions are wrong
  on plain, unformatted text -> the model overfit to dataset
  formatting/artifacts (e.g. Reuters datelines), not real content.

- If STEP 3 shows dateline/wire-service words as top features
  -> confirms the model is using superficial cues.

- Fix: strip datelines/outlet names from training text before
  vectorizing, and/or add more diverse real & fake sources so the
  model can't shortcut to "does this look like Reuters".
""")