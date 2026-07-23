# Fake News Detection System - Complete Local Training

A complete end-to-end machine learning pipeline: **data preprocessing → TF-IDF vectorization → Baseline RF training → PSO feature selection → PSO+RF training → Flask API → Interactive web interface**

## Project Structure

```
fake_news_project/
├── data/
│   ├── fake.csv          (fake news articles)
│   └── true.csv          (real news articles)
├── models/               (trained models will be saved here)
├── train_models.py       (training script)
├── app.py               (Flask backend)
├── index.html           (interactive frontend)
└── requirements.txt     (dependencies)
```

## Installation & Setup

### 1. **Create Project Folder**

```bash
mkdir fake_news_project
cd fake_news_project
mkdir data models
```

### 2. **Place Your CSV Files**

Copy your datasets to the `data/` folder:
- `fake.csv` — fake news articles (title, text, subject, date columns)
- `true.csv` — real news articles (title, text, subject, date columns)

### 3. **Install Dependencies**

```bash
pip install Flask Flask-CORS pandas numpy scikit-learn matplotlib seaborn Pillow
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. **Update Paths (if needed)**

Edit `train_models.py` and `app.py`, lines 7-8, with your actual username:

```python
DATA_PATH = r'C:\Users\YOUR_USERNAME\fake_news_project\data'
MODELS_PATH = r'C:\Users\YOUR_USERNAME\fake_news_project\models'
```

---

## Workflow

### **Step 1: Train All Models**

Run the training script to build both models from scratch:

```bash
python train_models.py
```

This will:
1. ✅ Load and combine fake.csv + true.csv
2. ✅ Sample 5,000 articles for training
3. ✅ Apply TF-IDF vectorization (300 features)
4. ✅ Train Baseline Random Forest (30 trees, depth 8)
5. ✅ Run Binary PSO for feature selection (20 iterations, 15 particles)
6. ✅ Train PSO+RF model (500 trees, unlimited depth)
7. ✅ Save all models to `models/` folder

**Expected output:**
```
==============================================================
FAKE NEWS DETECTION - COMPLETE TRAINING PIPELINE
==============================================================

[1/6] Loading datasets...
✅ Loaded 44898 total articles
   - Fake news: 23481
   - Real news: 21417
✅ Sampled 5000 articles for training
   - Training set: 4000
   - Test set: 1000

[2/6] Applying TF-IDF vectorization...
✅ Vectorized to 300 features

[3/6] Training Baseline Random Forest...
✅ Baseline RF trained!
   - Accuracy: 0.9933
   - AUC-ROC: 0.9988

[4/6] Running Binary PSO for feature selection...
   Iteration 5/20 - Best fitness: 0.9850
   ...
   Iteration 20/20 - Best fitness: 0.9917
✅ PSO completed!
   - Selected 143 features out of 300
   - Feature reduction: 52.3%

[5/6] Training PSO+RF model...
✅ PSO+RF trained!
   - Accuracy: 0.9917
   - AUC-ROC: 0.9985

[6/6] Saving models...
✅ Models saved!
   - rf_baseline.pkl
   - rf_pso.pkl
   - tfidf.pkl
   - selected_features.pkl (143 features)

============================================================
TRAINING SUMMARY
============================================================

Baseline Random Forest:
  Accuracy: 0.9933 (99.33%)
  AUC-ROC:  0.9988
  Features: 300

PSO+RF Model:
  Accuracy: 0.9917 (99.17%)
  AUC-ROC:  0.9985
  Features: 143 (52% reduction)

✅ All models trained and saved!
You can now run: python app.py
============================================================
```

**Time:** ~10-15 minutes (depends on your CPU)

---

### **Step 2: Start Flask Backend**

After training completes, run:

```bash
python app.py
```

You should see:
```
Loading datasets...
Dataset loaded: 44898 articles
Loading models...
   - Baseline features: 300
   - PSO selected features: 143
✅ All models loaded!
Generating evaluation plots...
✅ All plots generated!
 * Running on http://localhost:5000
```

**Leave this terminal open.**

---

### **Step 3: Open Frontend in Browser**

Open `index.html` in your browser, or navigate to:
```
http://localhost:5000
```

---

## What You Get

### 🏠 **Home Tab**
- Project overview
- Model statistics (Accuracy, AUC, Features)
- Dataset info

### 🎯 **Baseline RF Tab**
- Live text prediction
- Real-time results with confidence scores
- Fake/Real probability distribution

### 🚀 **PSO+RF Tab**
- Same interface as Baseline
- Compare results across models

### 📊 **Evaluation Tab**
- Confusion matrices (both models)
- ROC curves with AUC
- Feature importance (top 20)
- Metrics comparison

---

## Model Details

### **Baseline Random Forest**
- **Input:** TF-IDF (300 features, unigrams + bigrams)
- **Model:** Random Forest (30 trees, max_depth=8)
- **Purpose:** Simple baseline for comparison
- **Performance:** ~99.33% accuracy

### **PSO+RF (Proposed)**
- **Input:** TF-IDF (300 features)
- **Feature Selection:** Binary PSO (20 iterations, 15 particles)
  - Selects ~143 best features (52% reduction)
  - Evaluates each particle by training a test RF
- **Model:** Random Forest (500 trees, unlimited depth)
- **Trained On:** Only the 143 selected features
- **Performance:** ~99.17% accuracy with 52% fewer features
- **Benefit:** Faster inference, less overfitting, simpler model

---

## Troubleshooting

### **Training takes too long**

PSO evaluates many feature combinations. To speed up:
1. Reduce `n_iterations` in `train_models.py` (line ~220): `n_iterations=10`
2. Reduce `n_particles` (line ~220): `n_particles=10`
3. Sample fewer articles (line ~40): `n=2000`

### **Models don't exist when running app.py**

Make sure you ran `python train_models.py` first and it completed successfully.

### **Port 5000 already in use**

In `app.py`, change the last line:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### **Predictions failing**

Check Flask terminal for error messages. Verify:
- All CSV files are in `data/` folder
- All `.pkl` files were created in `models/` folder
- Paths in `train_models.py` and `app.py` are correct

---

## Files Generated After Training

After running `train_models.py`, you'll have in `models/`:

| File | Size | Purpose |
|------|------|---------|
| `rf_baseline.pkl` | ~10MB | Baseline Random Forest model |
| `rf_pso.pkl` | ~30MB | PSO+RF model (500 trees) |
| `tfidf.pkl` | ~1MB | TF-IDF vectorizer |
| `selected_features.pkl` | <1KB | Indices of 143 selected features |

---

## Performance Tips

1. **First run is slowest** — plots generate at startup, takes ~30 seconds
2. **Predictions are instant** — models cached in memory after startup
3. **Use GPU (optional)** — modify RandomForest `n_jobs=-1` to use all cores
4. **Batch predictions** — modify `/api/predict` to handle multiple texts

---

## Next Steps

### Option 1: Improve PSO
- Increase iterations: `n_iterations=50`
- More particles: `n_particles=30`
- Custom fitness function (precision, F1-score, etc.)

### Option 2: Try Other Algorithms
- Genetic Algorithm for feature selection
- Gradient Boosting instead of Random Forest
- Deep learning (LSTM, BERT)

### Option 3: Deploy
- Docker containerization
- Deploy to Heroku/Railway/Render
- Create a REST API for mobile apps

---

## Summary

**You now have:**
✅ Complete training pipeline from raw CSV to deployed models
✅ Baseline RF vs PSO+RF comparison
✅ Interactive web interface with live predictions
✅ All evaluation plots and metrics
✅ Full local control — no Colab needed

**Total time:** ~15 min training + setup = **30 minutes to production** 🚀
