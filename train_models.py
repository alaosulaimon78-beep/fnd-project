"""
FIXED training script: strips Reuters datelines and wire-service artifacts
from text BEFORE vectorization, so the model learns real content patterns
instead of superficial formatting cues.

KEY CHANGES:
1. Added clean_text() function that removes:
   - Reuters/AP/wire-service dateline patterns (CITY, STATE/CITY - ...)
   - Embedded video/image/via tags common in the ISOT dataset
   - URLs
   
2. Apply clean_text() to ALL articles before vectorizing

3. Rest of training stays the same

USAGE:
  python train_models_FIXED.py
"""

import pickle
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pyswarm import pso


def clean_text(text):
    """
    Strip Reuters-style datelines and wire-service artifacts that the model
    was overfitting to (e.g. 'CITY, STATE (Reuters) - ' at the start).
    
    This forces the model to learn actual content patterns instead of
    superficial formatting cues.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip Reuters/AP/wire-service dateline pattern
    # Matches: "WEST PALM BEACH, Fla./WASHINGTON (Reuters) - "
    text = re.sub(
        r'^[A-Z\s,/.]*\((?:Reuters|AP|Reuters|Associated Press)[^)]*\)\s*[-–]\s*',
        '',
        text,
        flags=re.IGNORECASE
    )
    
    # Remove embedded video/image/via tags (common in ISOT dataset)
    text = re.sub(r'\[image\]|\[video\]|via\s+\w+', '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


print("=" * 70)
print("FAKE NEWS DETECTION - TRAINING WITH CLEANED TEXT")
print("=" * 70)

# ---------------------------------------------------------------
# [1/7] Load datasets
# ---------------------------------------------------------------
print("\n[1/7] Loading datasets...")

fake_df = pd.read_csv('data/fake.csv')
true_df = pd.read_csv('data/true.csv')

print(f"✅ Loaded {len(fake_df)} fake articles")
print(f"✅ Loaded {len(true_df)} true articles")
print(f"✅ Total: {len(fake_df) + len(true_df)} articles")

# ---------------------------------------------------------------
# [2/7] Clean text (remove Reuters datelines & artifacts)
# ---------------------------------------------------------------
print("\n[2/7] Cleaning text (stripping Reuters datelines, etc.)...")

# Identify text column
text_col = 'text' if 'text' in fake_df.columns else fake_df.columns[0]

fake_df['text_cleaned'] = fake_df[text_col].apply(clean_text)
true_df['text_cleaned'] = true_df[text_col].apply(clean_text)

# Show before/after examples
print(f"\nExample BEFORE cleaning:")
print(f"  {fake_df[text_col].iloc[0][:150]}...")
print(f"\nExample AFTER cleaning:")
print(f"  {fake_df['text_cleaned'].iloc[0][:150]}...")

# ---------------------------------------------------------------
# [3/7] Set labels
# ---------------------------------------------------------------
print("\n[3/7] Setting labels...")

fake_df['label'] = 0  # Fake = 0
true_df['label'] = 1  # Real = 1

# Combine
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sample for faster training (optional - remove if you want full dataset)
sample_size = 5000
if len(df) > sample_size:
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"✅ Sampled {sample_size} articles for faster training")
else:
    df_sample = df

print(f"  - Fake: {(df_sample['label'] == 0).sum()}")
print(f"  - Real: {(df_sample['label'] == 1).sum()}")

# ---------------------------------------------------------------
# [4/7] Train/test split & vectorize
# ---------------------------------------------------------------
print("\n[4/7] Train/test split and TF-IDF vectorization...")

X_train, X_test, y_train, y_test = train_test_split(
    df_sample['text_cleaned'].values,
    df_sample['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=df_sample['label']
)

# Vectorize using CLEANED text
vectorizer = TfidfVectorizer(max_features=300)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Vectorized to {X_train_vec.shape[1]} features")
print(f"  - Training set: {X_train_vec.shape[0]}")
print(f"  - Test set: {X_test_vec.shape[0]}")

# ---------------------------------------------------------------
# [5/7] Train baseline Random Forest
# ---------------------------------------------------------------
print("\n[5/7] Training Baseline Random Forest...")

rf_baseline = RandomForestClassifier(
    n_estimators=30,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train_vec, y_train)

acc_baseline = rf_baseline.score(X_test_vec, y_test)
print(f"✅ Baseline RF trained!")
print(f"  - Test accuracy: {acc_baseline:.4f}")

# ---------------------------------------------------------------
# [6/7] Binary PSO for feature selection
# ---------------------------------------------------------------
print("\n[6/7] Running Binary PSO for feature selection...")

def fitness(selected):
    """Fitness function for PSO: maximize accuracy on selected features."""
    # PSO returns continuous values, so threshold at 0.5 to get binary selection
    selected_binary = (selected > 0.5).astype(int)
    
    if selected_binary.sum() == 0:
        return 1.0  # penalize empty selection (return high value to minimize)
    
    rf_temp = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    rf_temp.fit(X_train_vec[:, selected_binary == 1], y_train)
    accuracy = rf_temp.score(X_test_vec[:, selected_binary == 1], y_test)
    return 1.0 - accuracy  # return error (1 - accuracy) to minimize

# PSO with fewer iterations for speed
result = pso(
    fitness,
    lb=np.zeros(X_train_vec.shape[1]),
    ub=np.ones(X_train_vec.shape[1]),
    maxiter=20,
    swarmsize=10
)
xopt = result[0]  # Extract just the optimal parameters

selected_features = np.where(xopt > 0.5)[0]  # threshold at 0.5
print(f"✅ PSO completed!")
print(f"  - Selected {len(selected_features)} features out of {X_train_vec.shape[1]}")
print(f"  - Feature reduction: {100 * (1 - len(selected_features) / X_train_vec.shape[1]):.1f}%")

# ---------------------------------------------------------------
# [7/7] Train PSO+RF model
# ---------------------------------------------------------------
print("\n[7/7] Training PSO+RF model...")

rf_pso = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_pso.fit(X_train_vec[:, selected_features], y_train)

acc_pso = rf_pso.score(X_test_vec[:, selected_features], y_test)
print(f"✅ PSO+RF trained!")
print(f"  - Test accuracy: {acc_pso:.4f}")

# ---------------------------------------------------------------
# Save models
# ---------------------------------------------------------------
print("\n[8/8] Saving models...")

with open('models/rf_baseline.pkl', 'wb') as f:
    pickle.dump(rf_baseline, f)

with open('models/rf_pso.pkl', 'wb') as f:
    pickle.dump(rf_pso, f)

with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("✅ Models saved!")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"""
Baseline Random Forest:
  Accuracy: {acc_baseline:.4f}
  Features: 300

PSO+RF Model:
  Accuracy: {acc_pso:.4f}
  Features: {len(selected_features)} ({100 * len(selected_features) / X_train_vec.shape[1]:.1f}%)

KEY CHANGE: Training text was cleaned to remove Reuters datelines
and wire-service artifacts BEFORE vectorization. This forces the
model to learn real content patterns instead of superficial formatting.

✅ All models trained and saved!
You can now run: python app.py
""")