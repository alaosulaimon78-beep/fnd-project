import warnings
warnings.filterwarnings('ignore')

import os

import matplotlib
matplotlib.use('Agg')

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
CORS(app)

# Paths — relative to this file, so it works both locally and on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Load data
print("Loading datasets...")
fake_df = pd.read_csv(os.path.join(DATA_PATH, 'fake.csv'))
true_df = pd.read_csv(os.path.join(DATA_PATH, 'true.csv'))

fake_df['label'] = 0  # Fake = 0
true_df['label'] = 1  # Real = 1
df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Combine title + text
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

print(f"Dataset loaded: {len(df)} articles")

# Load models
print("Loading models...")
with open(os.path.join(MODELS_PATH, 'rf_baseline.pkl'), 'rb') as f:
    rf_baseline = pickle.load(f)
with open(os.path.join(MODELS_PATH, 'rf_pso.pkl'), 'rb') as f:
    rf_pso = pickle.load(f)
with open(os.path.join(MODELS_PATH, 'tfidf.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)
with open(os.path.join(MODELS_PATH, 'selected_features.pkl'), 'rb') as f:
    selected_features = pickle.load(f)

print("✅ All models loaded!")
print(f"   - Baseline features: 300")
print(f"   - PSO selected features: {len(selected_features)}")

# Generate evaluation plots at startup
print("Generating evaluation plots...")

X = vectorizer.transform(df['content'])
y = df['label'].values

# Baseline predictions (all 300 features)
y_pred_baseline = rf_baseline.predict(X)
y_proba_baseline = rf_baseline.predict_proba(X)[:, 1]

# PSO+RF predictions (selected features only)
X_pso = X[:, selected_features]
y_pred_pso = rf_pso.predict(X_pso)
y_proba_pso = rf_pso.predict_proba(X_pso)[:, 1]

# Store for later use
evaluation_data = {
    'X': X,
    'y': y,
    'y_pred_baseline': y_pred_baseline,
    'y_proba_baseline': y_proba_baseline,
    'y_pred_pso': y_pred_pso,
    'y_proba_pso': y_proba_pso
}

def generate_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def get_confusion_matrix_plot(y_true, y_pred, title):
    """Generate confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    ax.set_xticklabels(['Fake', 'Real'])
    ax.set_yticklabels(['Fake', 'Real'])
    return generate_plot_base64(fig)

def get_roc_curve_plot(y_true, y_proba, title):
    """Generate ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return generate_plot_base64(fig)

def get_feature_importance_plot(model, feature_names, title, top_n=20):
    """Generate feature importance plot"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    return generate_plot_base64(fig)

def get_metrics_comparison_plot():
    """Generate metrics comparison plot"""
    acc_baseline = accuracy_score(evaluation_data['y'], evaluation_data['y_pred_baseline'])
    acc_pso = accuracy_score(evaluation_data['y'], evaluation_data['y_pred_pso'])

    fpr_b, tpr_b, _ = roc_curve(evaluation_data['y'], evaluation_data['y_proba_baseline'])
    auc_b = auc(fpr_b, tpr_b)

    fpr_p, tpr_p, _ = roc_curve(evaluation_data['y'], evaluation_data['y_proba_pso'])
    auc_p = auc(fpr_p, tpr_p)

    metrics = ['Accuracy', 'AUC']
    baseline_scores = [acc_baseline, auc_b]
    pso_scores = [acc_pso, auc_p]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, baseline_scores, width, label='Baseline RF', color='steelblue')
    ax.bar(x + width/2, pso_scores, width, label='PSO+RF', color='darkorange')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0.95, 1.01])
    ax.grid(axis='y', alpha=0.3)
    return generate_plot_base64(fig)

# Generate plots at startup
print("Generating plots...")

# For PSO feature importance, we need feature names of selected features
feature_names_all = vectorizer.get_feature_names_out()
feature_names_pso = feature_names_all[selected_features]

plots = {
    'cm_baseline': get_confusion_matrix_plot(evaluation_data['y'], evaluation_data['y_pred_baseline'], 'Confusion Matrix - Baseline RF'),
    'cm_pso': get_confusion_matrix_plot(evaluation_data['y'], evaluation_data['y_pred_pso'], 'Confusion Matrix - PSO+RF'),
    'roc_baseline': get_roc_curve_plot(evaluation_data['y'], evaluation_data['y_proba_baseline'], 'ROC Curve - Baseline RF'),
    'roc_pso': get_roc_curve_plot(evaluation_data['y'], evaluation_data['y_proba_pso'], 'ROC Curve - PSO+RF'),
    'metrics_comparison': get_metrics_comparison_plot(),
    'feature_imp_baseline': get_feature_importance_plot(rf_baseline, feature_names_all, 'Top 20 Features - Baseline RF'),
    'feature_imp_pso': get_feature_importance_plot(rf_pso, feature_names_pso, 'Top 20 Features - PSO+RF')
}

print("✅ All plots generated!")

# Calculate stats
acc_baseline = accuracy_score(evaluation_data['y'], evaluation_data['y_pred_baseline'])
acc_pso = accuracy_score(evaluation_data['y'], evaluation_data['y_pred_pso'])

fpr_b, tpr_b, _ = roc_curve(evaluation_data['y'], evaluation_data['y_proba_baseline'])
auc_b = auc(fpr_b, tpr_b)

fpr_p, tpr_p, _ = roc_curve(evaluation_data['y'], evaluation_data['y_proba_pso'])
auc_p = auc(fpr_p, tpr_p)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return model statistics"""
    return jsonify({
        'baseline': {
            'accuracy': round(acc_baseline, 4),
            'auc': round(auc_b, 4),
            'features': 300
        },
        'pso': {
            'accuracy': round(acc_pso, 4),
            'auc': round(auc_p, 4),
            'features': len(selected_features)
        }
    })

@app.route('/api/plots', methods=['GET'])
def get_plots():
    """Return all evaluation plots"""
    return jsonify(plots)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on user input with confidence thresholds"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize
    X_test = vectorizer.transform([text])

    # ===== BASELINE MODEL =====
    pred_baseline = rf_baseline.predict(X_test)[0]
    proba_baseline = rf_baseline.predict_proba(X_test)[0]
    confidence_baseline = max(proba_baseline) * 100

    # Determine prediction with confidence threshold
    if confidence_baseline < 60:
        prediction_baseline = "UNCERTAIN"
        status_baseline = "⚠️ Low Confidence"
        message_baseline = "The model is uncertain about this prediction. Please review manually or provide more context."
    else:
        prediction_baseline = 'Real' if pred_baseline == 1 else 'Fake'
        status_baseline = "✅ Confident" if confidence_baseline > 75 else "⚠️ Moderate Confidence"
        message_baseline = ""

    # ===== PSO MODEL =====
    X_test_pso = X_test[:, selected_features]
    pred_pso = rf_pso.predict(X_test_pso)[0]
    proba_pso = rf_pso.predict_proba(X_test_pso)[0]
    confidence_pso = max(proba_pso) * 100

    # Determine prediction with confidence threshold
    if confidence_pso < 60:
        prediction_pso = "UNCERTAIN"
        status_pso = "⚠️ Low Confidence"
        message_pso = "The model is uncertain about this prediction. Please review manually or provide more context."
    else:
        prediction_pso = 'Real' if pred_pso == 1 else 'Fake'
        status_pso = "✅ Confident" if confidence_pso > 75 else "⚠️ Moderate Confidence"
        message_pso = ""

    return jsonify({
        'baseline': {
            'prediction': prediction_baseline,
            'confidence': round(confidence_baseline, 2),
            'status': status_baseline,
            'message': message_baseline,
            'fake_prob': round(proba_baseline[0] * 100, 2),
            'real_prob': round(proba_baseline[1] * 100, 2)
        },
        'pso': {
            'prediction': prediction_pso,
            'confidence': round(confidence_pso, 2),
            'status': status_pso,
            'message': message_pso,
            'fake_prob': round(proba_pso[0] * 100, 2),
            'real_prob': round(proba_pso[1] * 100, 2)
        },
        'disclaimer': {
            'title': '⚠️ Model Limitations',
            'description': 'This model was trained on formal news articles from 2016-2018 (Reuters, AP, etc.) and may not work well on:',
            'limitations': [
                '❌ Social media posts, tweets, or informal text',
                '❌ Modern news from 2024-2026 (may have different writing styles)',
                '❌ News from non-English sources or translations',
                '❌ Highly opinionated or satirical content',
                '❌ Visual misinformation or deepfakes'
            ],
            'recommendation': 'Use this tool as a first pass only. Always verify with fact-checking websites like Snopes, FactCheck.org, or PolitiFact for important decisions.'
        }
    })

@app.route('/')
def home():
    """Serve frontend"""
    try:
        index_path = os.path.join(BASE_DIR, 'index.html')
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading page: {str(e)}</h1>", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*70)
    print("✅ FLASK SERVER RUNNING")
    print("="*70)
    print(f"\n🌐 Listening on port {port}")
    print("\n📊 Features:")
    print("   - Home: Project overview & statistics")
    print("   - Baseline RF: Predictions with baseline model")
    print("   - PSO+RF: Predictions with optimized model")
    print("   - Evaluation: All plots and metrics")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
