"""
app.py - FraudSentinel AI
Routes:
  GET  /              -> Investigation dashboard
  GET  /results       -> Evaluation results & charts
  POST /api/analyze   -> ML + RAG + explainability
  GET  /api/metrics   -> Model metrics
  GET  /api/samples   -> Sample transactions
"""
import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

from explainer import explain_prediction
from features import prepare_single_record
from rag_engine import RAGRetriever

app = Flask(__name__)

MODEL = None
FEATURES = []
OPT_THRESHOLD = 0.5
RAG = None
ARTIFACT_ERROR = None
CATEGORY_MAPS = None
METRICS = {
    'model_name': 'Unavailable',
    'roc_auc': 0.0,
    'avg_precision': 0.0,
    'f1': 0.0,
    'recall': 0.0,
    'cv_mean': 0.0,
    'cv_std': 0.0,
    'opt_threshold': 0.5,
    'n_features': 0,
    'comparison_table': [],
    'smote_used': False,
    'shap_available': False,
    'cv_folds': 5,
    'n_train_raw': 0,
    'n_train_resampled': 0,
    'n_test': 0,
    'training_notes': [],
}

FALLBACK_CATEGORY_MAPS = {
    'card4': {'discover': 0, 'mastercard': 1, 'american express': 2, 'visa': 3, 'unknown': 4},
    'card6': {'charge card': 0, 'debit': 1, 'credit': 2, 'unknown': 3},
    'ProductCD': {'C': 0, 'H': 1, 'R': 2, 'S': 3, 'W': 4},
    'm_cols': [f'M{i}' for i in range(1, 10)],
    'm_mappings': {
        f'M{i}': {'F': 0, 'unknown': 1, 'T': 2, 'M0': 3, 'M2': 4} for i in range(1, 10)
    },
}

SAMPLES = [
    {
        'id': 'TXN-2987654',
        'display_merchant': 'Anonymous Online Store',
        'display_location': 'Unknown / Proxy IP',
        'display_time': '02:17 AM',
        'display_mode': 'Card-Not-Present',
        'TransactionAmt': 4897.00,
        'TransactionDT': 7620,
        'ProductCD': 'W',
        'card4': 'visa',
        'card6': 'debit',
        'P_emaildomain': 'anonymous.com',
        'R_emaildomain': 'gmail.com',
        'addr1': 300,
        'addr2': 45,
        **{f'C{i}': 14 for i in range(1, 15)},
        **{f'D{i}': 120.0 for i in range(1, 6)},
        **{f'M{i}': 'F' for i in range(1, 10)},
        **{f'V{i}': round(float(np.random.default_rng(i).uniform(1.5, 4.5)), 4) for i in range(1, 21)},
    },
    {
        'id': 'TXN-1234501',
        'display_merchant': 'Streaming Subscription',
        'display_location': 'Austin, TX',
        'display_time': '02:15 PM',
        'display_mode': 'Saved Card',
        'TransactionAmt': 19.99,
        'TransactionDT': 51300,
        'ProductCD': 'H',
        'card4': 'mastercard',
        'card6': 'credit',
        'P_emaildomain': 'outlook.com',
        'R_emaildomain': 'outlook.com',
        'addr1': 225,
        'addr2': 225,
        **{f'C{i}': 1 for i in range(1, 15)},
        **{f'D{i}': 200.0 for i in range(1, 6)},
        **{f'M{i}': 'T' for i in range(1, 10)},
        **{f'V{i}': round(float(np.random.default_rng(i + 100).uniform(0.0, 0.8)), 4) for i in range(1, 21)},
    },
    {
        'id': 'TXN-5548821',
        'display_merchant': 'Cash Transfer Service',
        'display_location': 'Unknown',
        'display_time': '11:58 PM',
        'display_mode': 'Wire Transfer',
        'TransactionAmt': 1200.00,
        'TransactionDT': 82680,
        'ProductCD': 'C',
        'card4': 'discover',
        'card6': 'debit',
        'P_emaildomain': 'mail.com',
        'R_emaildomain': 'yahoo.com',
        'addr1': 180,
        'addr2': 95,
        **{f'C{i}': 19 for i in range(1, 15)},
        **{f'D{i}': 5.0 for i in range(1, 6)},
        **{f'M{i}': 'F' for i in range(1, 10)},
        **{f'V{i}': round(float(np.random.default_rng(i + 200).uniform(1.0, 3.5)), 4) for i in range(1, 21)},
    },
]


def _append_training_note(message: str):
    notes = METRICS.setdefault('training_notes', [])
    if message not in notes:
        notes.append(message)


def load_artifacts():
    global MODEL, METRICS, FEATURES, OPT_THRESHOLD, CATEGORY_MAPS, RAG, ARTIFACT_ERROR
    try:
        with open('models/fraud_model.pkl', 'rb') as f:
            MODEL = pickle.load(f)
        with open('models/metrics.json') as f:
            METRICS = json.load(f)
        with open('models/features.json') as f:
            FEATURES = json.load(f)
        with open('models/threshold.json') as f:
            OPT_THRESHOLD = json.load(f)['threshold']

        category_maps_path = 'models/category_maps.json'
        if os.path.exists(category_maps_path):
            with open(category_maps_path) as f:
                CATEGORY_MAPS = json.load(f)
        else:
            CATEGORY_MAPS = FALLBACK_CATEGORY_MAPS
            _append_training_note(
                'Using fallback category maps. Re-run train_model.py to persist exact training mappings.'
            )

        RAG = RAGRetriever()
        ARTIFACT_ERROR = None
    except Exception as exc:
        MODEL = None
        FEATURES = []
        CATEGORY_MAPS = FALLBACK_CATEGORY_MAPS
        RAG = None
        ARTIFACT_ERROR = str(exc)
        _append_training_note(f'Artifacts unavailable: {ARTIFACT_ERROR}')


load_artifacts()


@app.route('/')
def index():
    return render_template(
        'index.html',
        metrics=METRICS,
        threshold=OPT_THRESHOLD,
        model_ready=MODEL is not None and RAG is not None,
        artifact_error=ARTIFACT_ERROR,
    )


@app.route('/results')
def results():
    return render_template(
        'results.html',
        metrics=METRICS,
        model_ready=MODEL is not None and RAG is not None,
        artifact_error=ARTIFACT_ERROR,
    )


@app.route('/api/samples')
def get_samples():
    return jsonify(SAMPLES)


@app.route('/api/metrics')
def get_metrics():
    return jsonify(METRICS)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if MODEL is None or RAG is None or not FEATURES:
        return jsonify({
            'error': 'Model artifacts are unavailable. Re-run train_model.py and restart the app.',
            'details': ARTIFACT_ERROR,
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        vec, prepared = prepare_single_record(data, FEATURES, category_maps=CATEGORY_MAPS)
        ml_prob = float(MODEL.predict_proba(vec.reshape(1, -1))[0][1])
        similar = RAG.retrieve(vec, top_k=3)
        rag_prob = RAG.rag_adjusted_score(ml_prob, similar, weight=0.15)

        feat_dict = {feature: float(prepared.get(feature, -999)) for feature in FEATURES}
        result = explain_prediction(
            feat_dict,
            ml_prob,
            rag_prob,
            similar,
            METRICS.get('feature_importances', {}),
            opt_threshold=OPT_THRESHOLD,
        )
        return jsonify({
            'txn_id': data.get('id', 'TXN-?'),
            'amount': data.get('TransactionAmt', 0),
            **result,
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    debug_enabled = os.getenv('FLASK_DEBUG', '').strip().lower() in {'1', 'true', 'yes'}
    port = int(os.getenv('PORT', '5000'))
    print(f"\nFraudSentinel AI (MSc) -> http://localhost:{port}")
    print(f"   Model     : {METRICS['model_name']}")
    print(f"   ROC-AUC   : {METRICS['roc_auc']}")
    print(f"   Threshold : {OPT_THRESHOLD:.3f} (auto-tuned)")
    if ARTIFACT_ERROR:
        print(f"   Warning   : {ARTIFACT_ERROR}")
    print()
    app.run(debug=debug_enabled, port=port)
