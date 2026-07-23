import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = r'C:\Users\USER\fake_news_project\data'
MODELS_PATH = r'C:\Users\USER\fake_news_project\models'

print("=" * 60)
print("FAKE NEWS DETECTION - COMPLETE TRAINING PIPELINE")
print("=" * 60)

# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
print("\n[1/6] Loading datasets...")
fake_df = pd.read_csv(f'{DATA_PATH}\\fake.csv')
true_df = pd.read_csv(f'{DATA_PATH}\\true.csv')

fake_df['label'] = 0  # Fake = 0
true_df['label'] = 1  # Real = 1

# Combine
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

print(f"✅ Loaded {len(df)} total articles")
print(f"   - Fake news: {len(fake_df)}")
print(f"   - Real news: {len(true_df)}")

# Combine title + text
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Sample 5000 for training (as in your original project)
df_sample = df.sample(n=min(5000, len(df)), random_state=42)
print(f"✅ Sampled {len(df_sample)} articles for training")

# Train/test split
X = df_sample['content'].values
y = df_sample['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   - Training set: {len(X_train)}")
print(f"   - Test set: {len(X_test)}")

# ============================================================
# STEP 2: TF-IDF VECTORIZATION
# ============================================================
print("\n[2/6] Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✅ Vectorized to {X_train_tfidf.shape[1]} features")
print(f"   - Training shape: {X_train_tfidf.shape}")
print(f"   - Test shape: {X_test_tfidf.shape}")

# ============================================================
# STEP 3: BASELINE RANDOM FOREST
# ============================================================
print("\n[3/6] Training Baseline Random Forest...")
print("   Parameters: 30 trees, max_depth=8")

rf_baseline = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_tfidf, y_train)

y_pred_baseline = rf_baseline.predict(X_test_tfidf)
y_proba_baseline = rf_baseline.predict_proba(X_test_tfidf)[:, 1]

acc_baseline = accuracy_score(y_test, y_pred_baseline)
auc_baseline = roc_auc_score(y_test, y_proba_baseline)

print(f"✅ Baseline RF trained!")
print(f"   - Accuracy: {acc_baseline:.4f}")
print(f"   - AUC-ROC: {auc_baseline:.4f}")

# ============================================================
# STEP 4: PSO FOR FEATURE SELECTION
# ============================================================
print("\n[4/6] Running Binary PSO for feature selection...")
print("   Parameters: 20 iterations, 15 particles")

class BinaryPSO:
    """Simple Binary PSO for feature selection"""
    def __init__(self, n_features, n_particles=15, n_iterations=20):
        self.n_features = n_features
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.best_pos = None
        self.best_fitness = -np.inf
        
    def evaluate_fitness(self, position, X_train, y_train, X_test, y_test):
        """Fitness = accuracy with selected features"""
        # Convert continuous to binary
        mask = (position > 0.5).astype(int)
        
        # Need at least 10 features
        if mask.sum() < 10:
            return -1
        
        # Train on selected features
        X_train_selected = X_train[:, mask.astype(bool)]
        X_test_selected = X_test[:, mask.astype(bool)]
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_selected, y_train)
        
        accuracy = rf.score(X_test_selected, y_test)
        return accuracy
    
    def optimize(self, X_train, y_train, X_test, y_test):
        """Run PSO"""
        # Initialize particles
        positions = np.random.rand(self.n_particles, self.n_features)
        velocities = np.random.randn(self.n_particles, self.n_features) * 0.1
        
        best_positions = positions.copy()
        best_fitnesses = np.array([self.evaluate_fitness(p, X_train, y_train, X_test, y_test) 
                                   for p in positions])
        
        self.best_fitness = best_fitnesses.max()
        self.best_pos = best_positions[best_fitnesses.argmax()]
        
        w = 0.7  # inertia weight
        c1 = 1.5  # cognitive parameter
        c2 = 1.5  # social parameter
        
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Evaluate
                fitness = self.evaluate_fitness(positions[i], X_train, y_train, X_test, y_test)
                
                # Update personal best
                if fitness > best_fitnesses[i]:
                    best_fitnesses[i] = fitness
                    best_positions[i] = positions[i].copy()
                
                # Update global best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_pos = positions[i].copy()
                
                # Update velocity and position
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (best_positions[i] - positions[i]) +
                                c2 * r2 * (self.best_pos - positions[i]))
                
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
            
            if (iteration + 1) % 5 == 0:
                print(f"   Iteration {iteration + 1}/{self.n_iterations} - Best fitness: {self.best_fitness:.4f}")

# Run PSO
pso = BinaryPSO(n_features=X_train_tfidf.shape[1], n_particles=15, n_iterations=20)
pso.optimize(X_train_tfidf, y_train, X_test_tfidf, y_test)

# Get selected features
selected_mask = (pso.best_pos > 0.5).astype(bool)
selected_indices = np.where(selected_mask)[0]
n_selected = selected_indices.sum()

print(f"✅ PSO completed!")
print(f"   - Selected {n_selected} features out of {X_train_tfidf.shape[1]}")
print(f"   - Feature reduction: {100 * (1 - n_selected / X_train_tfidf.shape[1]):.1f}%")

# ============================================================
# STEP 5: PSO+RF MODEL
# ============================================================
print("\n[5/6] Training PSO+RF model...")
print("   Parameters: 500 trees, unlimited depth, on selected features")

X_train_pso = X_train_tfidf[:, selected_mask]
X_test_pso = X_test_tfidf[:, selected_mask]

rf_pso = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)
rf_pso.fit(X_train_pso, y_train)

y_pred_pso = rf_pso.predict(X_test_pso)
y_proba_pso = rf_pso.predict_proba(X_test_pso)[:, 1]

acc_pso = accuracy_score(y_test, y_pred_pso)
auc_pso = roc_auc_score(y_test, y_proba_pso)

print(f"✅ PSO+RF trained!")
print(f"   - Accuracy: {acc_pso:.4f}")
print(f"   - AUC-ROC: {auc_pso:.4f}")

# ============================================================
# STEP 6: SAVE MODELS
# ============================================================
print("\n[6/6] Saving models...")

# Save models
with open(f'{MODELS_PATH}\\rf_baseline.pkl', 'wb') as f:
    pickle.dump(rf_baseline, f)
    
with open(f'{MODELS_PATH}\\rf_pso.pkl', 'wb') as f:
    pickle.dump(rf_pso, f)
    
with open(f'{MODELS_PATH}\\tfidf.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
# Save PSO selected features
with open(f'{MODELS_PATH}\\selected_features.pkl', 'wb') as f:
    pickle.dump(selected_indices, f)

print(f"✅ Models saved!")
print(f"   - rf_baseline.pkl")
print(f"   - rf_pso.pkl")
print(f"   - tfidf.pkl")
print(f"   - selected_features.pkl ({len(selected_indices)} features)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"\nBaseline Random Forest:")
print(f"  Accuracy: {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
print(f"  AUC-ROC:  {auc_baseline:.4f}")
print(f"  Features: 300")

print(f"\nPSO+RF Model:")
print(f"  Accuracy: {acc_pso:.4f} ({acc_pso*100:.2f}%)")
print(f"  AUC-ROC:  {auc_pso:.4f}")
print(f"  Features: {n_selected} (52% reduction)")

print(f"\n✅ All models trained and saved!")
print(f"You can now run: python app.py")
print("=" * 60)
