"""
features.py
===========
Feature engineering for IEEE-CIS transaction data.
Uses persisted categorical mappings so training and inference stay aligned.
"""
import math
import re

import numpy as np
import pandas as pd

HIGH_RISK_DOMAINS = {'anonymous.com', 'mail.com', 'gmail.com', 'yahoo.com', 'hotmail.com'}
CATEGORICAL_DEFAULTS = {
    'card4': 'unknown',
    'card6': 'unknown',
    'ProductCD': 'W',
}


def _normalize_text(value, default='unknown') -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _build_mapping(series: pd.Series, default='unknown') -> dict:
    normalized = sorted({_normalize_text(value, default) for value in series})
    return {value: idx for idx, value in enumerate(normalized)}


def _encode_with_mapping(value, mapping: dict, default='unknown') -> int:
    normalized = _normalize_text(value, default)
    if normalized in mapping:
        return mapping[normalized]
    unknown_key = default if default in mapping else next(iter(mapping), default)
    return mapping.get(unknown_key, 0)


def transform_frame(df: pd.DataFrame, category_maps: dict | None = None):
    feat = df.copy()

    # Time features
    feat['hour'] = (feat['TransactionDT'] // 3600) % 24
    feat['day_of_week'] = (feat['TransactionDT'] // (3600 * 24)) % 7
    feat['is_night'] = feat['hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)
    feat['is_weekend'] = feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Amount features
    feat['amt_log'] = np.log1p(feat['TransactionAmt'])
    feat['amt_cents'] = (feat['TransactionAmt'] % 1).round(2)
    feat['is_round_amt'] = (feat['TransactionAmt'] % 1 == 0).astype(int)

    # Email features
    feat['P_email_risk'] = feat['P_emaildomain'].apply(
        lambda x: 1 if _normalize_text(x, '') in HIGH_RISK_DOMAINS else 0
    )
    feat['email_match'] = (
        feat['P_emaildomain'].fillna('unk') == feat['R_emaildomain'].fillna('unk')
    ).astype(int)

    # Address match
    feat['addr_match'] = (
        feat['addr1'].fillna(-1) == feat['addr2'].fillna(-2)
    ).astype(int)

    if category_maps is None:
        category_maps = {
            'card4': _build_mapping(feat['card4'], default=CATEGORICAL_DEFAULTS['card4']),
            'card6': _build_mapping(feat['card6'], default=CATEGORICAL_DEFAULTS['card6']),
            'ProductCD': _build_mapping(feat['ProductCD'], default=CATEGORICAL_DEFAULTS['ProductCD']),
        }

        m_cols = sorted([c for c in feat.columns if re.match(r'^M\d+$', c)])
        category_maps['m_cols'] = m_cols
        category_maps['m_mappings'] = {
            m: _build_mapping(feat[m], default='unknown') for m in m_cols
        }
    else:
        category_maps = {
            **category_maps,
            'm_cols': category_maps.get('m_cols', []),
            'm_mappings': category_maps.get('m_mappings', {}),
        }

    feat['card_type_enc'] = feat['card4'].apply(
        lambda x: _encode_with_mapping(x, category_maps['card4'], CATEGORICAL_DEFAULTS['card4'])
    )
    feat['card_bank_enc'] = feat['card6'].apply(
        lambda x: _encode_with_mapping(x, category_maps['card6'], CATEGORICAL_DEFAULTS['card6'])
    )
    feat['product_enc'] = feat['ProductCD'].apply(
        lambda x: _encode_with_mapping(x, category_maps['ProductCD'], CATEGORICAL_DEFAULTS['ProductCD'])
    )

    for m in category_maps['m_cols']:
        mapping = category_maps['m_mappings'].get(m, {'unknown': 0})
        feat[m + '_enc'] = feat[m].apply(lambda x: _encode_with_mapping(x, mapping))

    return feat, category_maps


def engineer_features(df: pd.DataFrame):
    feat, category_maps = transform_frame(df)

    # C-columns
    c_cols = sorted([c for c in df.columns if re.match(r'^C\d+$', c)])

    # D-columns (first 5)
    d_cols = sorted([c for c in df.columns if re.match(r'^D\d+$', c)])[:5]

    # M-columns
    m_enc_cols = [m + '_enc' for m in category_maps['m_cols']]

    # V-columns — top 20 by correlation with isFraud
    v_cols_all = sorted([c for c in df.columns if re.match(r'^V\d+$', c)])
    if 'isFraud' in df.columns and len(v_cols_all) > 0:
        v_corr = feat[v_cols_all + ['isFraud']].corr()['isFraud'].abs()
        top_v = v_corr.drop('isFraud').nlargest(min(20, len(v_cols_all))).index.tolist()
    else:
        top_v = v_cols_all[:20]

    features = (
        ['TransactionAmt', 'amt_log', 'amt_cents', 'is_round_amt',
         'hour', 'day_of_week', 'is_night', 'is_weekend',
         'P_email_risk', 'email_match',
         'card_type_enc', 'card_bank_enc', 'product_enc', 'addr_match']
        + c_cols + d_cols + m_enc_cols + top_v
    )
    features = [f for f in features if f in feat.columns]

    X = feat[features].fillna(-999)
    y = feat['isFraud'] if 'isFraud' in feat.columns else None

    return X, y, features, category_maps


def prepare_single_record(data: dict, features: list, category_maps: dict | None = None):
    base = dict(data)
    dt = float(base.get('TransactionDT', 43200))
    amt = float(base.get('TransactionAmt', 0))

    base['TransactionDT'] = dt
    base['TransactionAmt'] = amt
    base['P_emaildomain'] = _normalize_text(base.get('P_emaildomain', ''), '')
    base['R_emaildomain'] = _normalize_text(base.get('R_emaildomain', ''), '')
    base['card4'] = _normalize_text(base.get('card4', CATEGORICAL_DEFAULTS['card4']), CATEGORICAL_DEFAULTS['card4'])
    base['card6'] = _normalize_text(base.get('card6', CATEGORICAL_DEFAULTS['card6']), CATEGORICAL_DEFAULTS['card6'])
    base['ProductCD'] = _normalize_text(base.get('ProductCD', CATEGORICAL_DEFAULTS['ProductCD']), CATEGORICAL_DEFAULTS['ProductCD'])

    for idx in range(1, 10):
        key = f'M{idx}'
        base[key] = _normalize_text(base.get(key, 'unknown'))

    frame = pd.DataFrame([base])
    transformed, _ = transform_frame(frame, category_maps=category_maps)
    record = transformed.iloc[0].to_dict()

    for feature in features:
        record.setdefault(feature, -999)

    vec = encode_single(record, features)
    return vec, record


def encode_single(data: dict, features: list) -> np.ndarray:
    vec = []
    for feature in features:
        value = data.get(feature, -999)
        if value is None or value == '':
            value = -999
        elif isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = -999
        elif isinstance(value, (float, np.floating)) and math.isnan(value):
            value = -999
        vec.append(float(value))
    return np.array(vec, dtype=np.float32)
