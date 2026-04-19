"""
explainer.py
============
Prediction explanation engine.
Falls back to feature-importance rules if SHAP metadata is unavailable.
"""
import json
import os

HIGH_RISK_DOMAINS = {'anonymous.com', 'mail.com', 'gmail.com', 'yahoo.com', 'hotmail.com'}

SHAP_META = {}
if os.path.exists('models/shap_values.json'):
    with open('models/shap_values.json') as f:
        SHAP_META = json.load(f)


def explain_prediction(features: dict, ml_prob: float, rag_prob: float,
                        similar_cases: list, feature_importances: dict,
                        opt_threshold: float = 0.5) -> dict:
    prob_pct = round(rag_prob * 100, 1)
    ml_prob_pct = round(ml_prob * 100, 1)
    thresh_pct = round(opt_threshold * 100, 1)

    if rag_prob >= opt_threshold + 0.2:
        risk_level = 'HIGH'
        recommendation = 'Block transaction immediately'
    elif rag_prob >= opt_threshold:
        risk_level = 'MEDIUM'
        recommendation = 'Flag for manual review'
    elif rag_prob >= opt_threshold - 0.15:
        risk_level = 'LOW-MEDIUM'
        recommendation = 'Approve with enhanced monitoring'
    else:
        risk_level = 'LOW'
        recommendation = 'Approve with routine monitoring'

    imp = SHAP_META.get('mean_shap', feature_importances)

    checks = [
        ('TransactionAmt',
         features.get('TransactionAmt', 0) > 500,
         f"High transaction amount ${features.get('TransactionAmt', 0):,.2f} above the review threshold"),
        ('is_night',
         features.get('is_night', 0) == 1,
         f"Night-time transaction at hour {int(features.get('hour', 0)):02d}:00"),
        ('P_email_risk',
         features.get('P_email_risk', 0) == 1,
         f"Purchaser email is on the high-risk domain list ({', '.join(sorted(HIGH_RISK_DOMAINS))})"),
        ('email_match',
         features.get('email_match', 1) == 0,
         'Purchaser and recipient email domains do not match'),
        ('addr_match',
         features.get('addr_match', 1) == 0,
         'Billing address fields do not match'),
        ('is_round_amt',
         features.get('is_round_amt', 0) == 1,
         'Round transaction amount matches a common scripted-fraud pattern'),
        ('is_weekend',
         features.get('is_weekend', 0) == 1,
         'Weekend transaction may fall into a lower-monitoring window'),
        ('amt_log',
         features.get('amt_log', 0) > 6.0,
         f"Log-scaled amount ({features.get('amt_log', 0):.2f}) indicates a large transaction"),
    ]

    risk_factors = []
    for feat_name, condition, msg in checks:
        if condition:
            risk_factors.append({
                'factor': msg,
                'importance': float(imp.get(feat_name, 0)),
                'shap_based': feat_name in SHAP_META.get('mean_shap', {}),
            })

    risk_factors.sort(key=lambda x: x['importance'], reverse=True)
    factor_texts = [r['factor'] for r in risk_factors]

    top = similar_cases[0] if similar_cases else None
    rag_ref = ''
    if top:
        rag_ref = (
            f" RAG retrieval found {len(similar_cases)} similar confirmed fraud case(s); "
            f"closest match {top['case_id']} at {top['similarity_pct']}% cosine similarity "
            f"(${top['amount']:,.2f}, {top['outcome']}). "
            f"RAG-adjusted score: {prob_pct}% (ML base: {ml_prob_pct}%)."
        )

    xai_note = 'SHAP TreeExplainer' if SHAP_META else 'Feature Importance'

    if risk_level == 'HIGH':
        summary = (
            f"Transaction classified as HIGH RISK with {prob_pct}% fraud probability "
            f"(threshold: {thresh_pct}%). {len(risk_factors)} significant risk factor(s) identified via {xai_note}."
            f"{rag_ref}"
        )
    elif risk_level == 'MEDIUM':
        summary = (
            f"Transaction shows moderate fraud signals with {prob_pct}% fraud probability "
            f"(threshold: {thresh_pct}%). {len(risk_factors)} flag(s) detected.{rag_ref} "
            f"Human review is recommended before processing."
        )
    else:
        summary = (
            f"Transaction appears lower risk with {prob_pct}% fraud probability "
            f"(threshold: {thresh_pct}%). {len(risk_factors)} minor flag(s) detected.{rag_ref}"
        )

    notes = []
    if features.get('is_night') and features.get('P_email_risk'):
        notes.append('Combined night transaction and high-risk email domain is a strong fraud signal.')
    if features.get('TransactionAmt', 0) > 1000 and features.get('is_round_amt'):
        notes.append('Large round-number amount can indicate scripted fraud or account takeover.')
    if not features.get('email_match') and not features.get('addr_match'):
        notes.append('Both email and address mismatches are present; verification is recommended before approval.')
    if ml_prob_pct != prob_pct:
        diff = round(prob_pct - ml_prob_pct, 1)
        direction = 'increased' if diff > 0 else 'decreased'
        notes.append(
            f"RAG context {direction} the fraud score by {abs(diff):.1f}% based on {len(similar_cases)} similar historical cases."
        )
    if SHAP_META:
        notes.append(f"Explainability powered by SHAP TreeExplainer; top feature: {SHAP_META.get('top_features', ['N/A'])[0]}.")
    if not notes:
        notes.append('No additional escalation notes at this time.')

    return {
        'risk_level': risk_level,
        'fraud_probability': prob_pct,
        'ml_probability': ml_prob_pct,
        'rag_adjusted': prob_pct != ml_prob_pct,
        'opt_threshold': thresh_pct,
        'recommendation': recommendation,
        'risk_factors': factor_texts,
        'evidence_summary': summary,
        'investigator_notes': notes,
        'similar_cases': similar_cases,
        'xai_method': xai_note,
    }
