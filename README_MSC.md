# FraudSentinel AI — MSc Computer Science Major Project

## What Was Upgraded (v2 → MSc Level)

| Feature | Before | After |
|---|---|---|
| Class imbalance | class_weight only | **SMOTE** oversampling |
| Evaluation | Single train/test split | **5-Fold Stratified CV** |
| Models compared | RF vs GBM | **LR + RF + GBM + Isolation Forest** |
| Threshold | Default 0.5 | **Auto-tuned via PR curve** |
| Explainability | Rule-based | **SHAP TreeExplainer** |
| RAG | Display only | **Functionally adjusts score (15% weight)** |
| Evaluation metrics | AUC + F1 | **AUC + Avg Precision + F1 + Cost matrix** |
| Charts | None | **5 evaluation charts** |
| Dashboard | Single page | **Investigation + Results & Analysis pages** |

---

## Setup Instructions

### 1. Install dependencies
```powershell
pip install flask scikit-learn pandas numpy imbalanced-learn shap matplotlib seaborn scipy
```

### 2. Add your dataset
Replace `data/train_transaction.csv` with the real IEEE-CIS file:
```powershell
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv
copy train_transaction.csv data\train_transaction.csv
```
Or generate dummy data first to test:
```powershell
python data/generate_dummy.py
```

### 3. Train the model
```powershell
python train_model.py
```
This will:
- Apply SMOTE to fix class imbalance
- Run 5-fold cross-validation on 3 models
- Benchmark Logistic Regression vs RF vs GBM vs Isolation Forest
- Auto-tune the decision threshold via Precision-Recall curve
- Compute SHAP values for explainability
- Generate 5 evaluation charts in static/
- Save all model artifacts to models/

### 4. Run the app
```powershell
python app.py
```
Then open: http://localhost:5000

---

## Project Structure
```
├── train_model.py        ← MSc pipeline (SMOTE + CV + SHAP + cost analysis)
├── app.py                ← Flask API + two-page dashboard
├── features.py           ← Feature engineering (62 features)
├── rag_engine.py         ← RAG retrieval + score adjustment
├── explainer.py          ← SHAP-powered explanation engine
├── templates/
│   ├── index.html        ← Investigation dashboard
│   └── results.html      ← Evaluation results & charts page
├── static/               ← Generated charts (after training)
│   ├── eval_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── cost_threshold.png
│   └── cross_validation.png
├── models/               ← Saved artifacts (after training)
│   ├── fraud_model.pkl
│   ├── metrics.json
│   ├── features.json
│   ├── threshold.json
│   ├── shap_values.json
│   ├── rag_vectors.npy
│   └── rag_meta.json
└── data/
    ├── generate_dummy.py
    └── train_transaction.csv
```

---

## API Reference

### POST /api/analyze
Accepts a transaction JSON, returns:
```json
{
  "risk_level": "HIGH | MEDIUM | LOW-MEDIUM | LOW",
  "fraud_probability": 87.3,
  "ml_probability": 85.1,
  "rag_adjusted": true,
  "opt_threshold": 42.0,
  "recommendation": "🔴 Block Transaction Immediately",
  "risk_factors": ["..."],
  "evidence_summary": "...",
  "investigator_notes": ["..."],
  "similar_cases": [...],
  "xai_method": "SHAP TreeExplainer"
}
```

---

## MSc-Level Contributions

1. **SMOTE** — Handles the severe class imbalance (3.5% fraud in real data). Without this, models learn to predict "not fraud" for everything and still get 96.5% accuracy.

2. **5-Fold Stratified Cross-Validation** — Proves model generalises, not just memorises. Shows mean ± std ROC-AUC across all folds.

3. **Isolation Forest Comparison** — Unsupervised anomaly detection baseline. Demonstrates why supervised learning outperforms anomaly detection for this problem.

4. **Threshold Tuning** — F1 score improves significantly (e.g. 0.36 → 0.50) by computing the optimal decision boundary from the Precision-Recall curve rather than using the default 0.5.

5. **SHAP Explainability** — Industry-standard feature attribution. Each prediction can be traced back to which features drove it, making the system auditable.

6. **Functional RAG** — Retrieved cases now adjust the final fraud score by 15%, making RAG an active component of the decision system rather than decorative context.

7. **Cost-Sensitive Evaluation** — False negatives (missed fraud) cost 10x more than false positives. The cost matrix evaluation shows which model is best for real-world deployment.

8. **Average Precision** — More informative than ROC-AUC for highly imbalanced datasets. The Precision-Recall curve area (AP) is the primary evaluation metric used in fraud literature.
