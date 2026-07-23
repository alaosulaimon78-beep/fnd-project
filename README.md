# Fake News Detection System - Flask + HTML/CSS/JS

A modern web application comparing Baseline Random Forest vs PSO+RF models for fake news detection.

## Project Structure

```
fake_news_project/
├── data/
│   ├── fake.csv          (fake news articles - label 0)
│   └── true.csv          (real news articles - label 1)
├── models/
│   ├── rf_baseline.pkl   (baseline random forest model)
│   ├── rf_pso.pkl        (PSO+RF model)
│   └── tfidf.pkl         (TF-IDF vectorizer)
├── app.py                (Flask backend)
├── index.html            (Frontend - open in browser)
└── requirements.txt      (Python dependencies)
```

## Installation & Setup

### 1. **Create Project Folder Structure**

In VS Code terminal:
```bash
# Create main project directory
mkdir fake_news_project
cd fake_news_project

# Create subdirectories
mkdir data
mkdir models
```

### 2. **Download Your Files**

- **From Google Drive**, download your `.pkl` files and place in `models/` folder:
  - `rf_baseline.pkl`
  - `rf_pso.pkl`
  - `tfidf.pkl`

- **Your CSV files** (fake.csv, true.csv) should already be in `data/` folder

### 3. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install Flask Flask-CORS pandas numpy scikit-learn matplotlib seaborn Pillow
```

### 4. **Verify Paths in `app.py`**

Open `app.py` and check these lines match your actual paths:

```python
DATA_PATH = r'C:\Users\USER\fake_news_project\data'
MODELS_PATH = r'C:\Users\USER\fake_news_project\models'
```

**Windows path example:**
```python
DATA_PATH = r'C:\Users\Sulaimon\fake_news_project\data'
MODELS_PATH = r'C:\Users\Sulaimon\fake_news_project\models'
```

## Running the Application

### Step 1: Start Flask Backend

In VS Code terminal (in the project folder):

```bash
python app.py
```

You should see output like:
```
Loading datasets...
Dataset loaded: XXXXX articles
Loading models...
✅ All models loaded!
Generating evaluation plots...
✅ All plots generated!
 * Running on http://localhost:5000
```

**Leave this terminal running.**

### Step 2: Open Frontend in Browser

- Open `index.html` directly in your browser (File → Open File → select index.html)
- Or right-click `index.html` in VS Code → "Open with Live Server"

You should see the Fake News Detection interface at `http://localhost:5000`

## Features

### 🏠 **Home Tab**
- Project overview
- Model statistics (Accuracy, AUC-ROC, Features)
- Dataset information

### 🎯 **Baseline RF Tab**
- Live text input for predictions
- Real-time results with confidence scores
- Fake/Real probability distribution

### 🚀 **PSO+RF Tab**
- Same prediction interface as Baseline
- Compare results between models

### 📊 **Evaluation Tab**
- Confusion matrices (both models)
- ROC curves with AUC scores
- Feature importance plots (top 20)
- Metrics comparison bar chart

## What Each Model Does

**Baseline Random Forest:**
- TF-IDF: 300 features (unigrams + bigrams)
- RF: 30 trees, depth 8
- Simple baseline for comparison

**PSO+RF (Proposed):**
- TF-IDF: 300 features
- Binary PSO: 20 iterations, 15 particles → selects ~96 features (52% reduction)
- RF: 500 trees, unlimited depth → trained on PSO-selected features
- Goal: Reduce features while maintaining accuracy

## Troubleshooting

### **Flask won't start**

Check:
- Python is installed: `python --version`
- All dependencies installed: `pip list`
- Paths in `app.py` are correct

### **Port 5000 already in use**

In `app.py`, change the last line:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change to 5001
```

Then open browser to: `http://localhost:5001`

### **Predictions not working**

- Ensure all `.pkl` files are in `models/` folder
- Ensure CSV files (fake.csv, true.csv) are in `data/` folder
- Check Flask terminal for error messages

### **Plots not loading**

- Plots generate when Flask starts — check Flask terminal for "✅ All plots generated!"
- Plots generate again when you click the "Evaluation" tab

## API Endpoints

Flask backend serves these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Model statistics (accuracy, AUC, feature count) |
| `/api/plots` | GET | All evaluation plots as base64 images |
| `/api/predict` | POST | Make prediction on text input |

## Tips for Better Results

1. **Longer text = better predictions** — single words may be less reliable
2. **Both models should agree** — if they disagree, check the confidence scores
3. **Evaluation plots** — generated from full training dataset, so very representative

## Deployment (Optional)

To deploy publicly, consider:
- **Vercel/Railway** (for frontend)
- **Render/Heroku** (for Flask backend)
- **Hugging Face Spaces** (free deployment)

## Notes

- Frontend uses CORS — make sure Flask-CORS is installed
- Models load once at startup — very fast predictions after that
- Plots cache in memory — no re-computation during session
- All preprocessing happens server-side

---

**Issues?** Check Flask terminal for detailed error messages and adjust paths accordingly.
