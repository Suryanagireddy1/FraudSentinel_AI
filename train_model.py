"""
train_model.py  —  FraudSentinel AI  (MSc Level)
Upgrades: SMOTE, 5-fold CV, Isolation Forest, threshold tuning,
SHAP, cost-sensitive evaluation, PR/ROC curves, RAG store fix.
"""
import os, json, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                              confusion_matrix, precision_recall_curve, roc_curve,
                              average_precision_score)
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)
from features import engineer_features

DARK='#050a14'; CYAN='#00e5ff'; RED='#ff2d55'; GREEN='#30d158'; ORANGE='#ff9f0a'; PURPLE='#bf5af2'
COST_FN=10; COST_FP=1; CV_FOLDS=5; MAX_RAG=2000
plt.rcParams.update({'figure.facecolor':DARK,'axes.facecolor':'#0a1020','axes.edgecolor':'#1e2a3a',
    'text.color':'#e0e0e0','axes.labelcolor':'#e0e0e0','xtick.color':'#5a6070',
    'ytick.color':'#5a6070','grid.color':'#1e2a3a','grid.alpha':0.5,'font.family':'monospace'})

# 1. LOAD
print("\n"+"="*60)
print("  FraudSentinel AI — MSc Training Pipeline")
print("="*60)
print("\n📂 Loading: data/train_transaction.csv")
df = pd.read_csv('data/train_transaction.csv')
print(f"   Rows: {len(df):,}  |  Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")

# 2. FEATURES
print("\n🔧 Engineering features...")
X, y, FEATURES, CATEGORY_MAPS = engineer_features(df)
print(f"   Total features: {len(FEATURES)}")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train_raw):,}  |  Test: {len(X_test):,}")

# 3. SMOTE
print("\n⚖️  Applying SMOTE...")
TRAINING_NOTES = []
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_raw, y_train_raw)
    print(f"   Before: Fraud={y_train_raw.sum():,}  After: Fraud={y_train_resampled.sum():,}")
    SMOTE_USED = True
except ImportError:
    print("   ⚠️  imbalanced-learn not installed — using class_weight='balanced'")
    X_train_resampled, y_train_resampled = X_train_raw, y_train_raw
    SMOTE_USED = False
    TRAINING_NOTES.append("imbalanced-learn missing; used class_weight='balanced' instead of SMOTE.")

cw = None if SMOTE_USED else 'balanced'


def build_pipeline(clf):
    steps = [('scaler', StandardScaler())]
    if SMOTE_USED:
        steps.append(('sampler', SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)))
        steps.append(('clf', clf))
        return ImbPipeline(steps)
    steps.append(('clf', clf))
    return Pipeline(steps)

# 4. MODELS
MODELS = {
    'Logistic Regression (Baseline)': build_pipeline(
        LogisticRegression(class_weight=cw, max_iter=1000, random_state=42)
    ),
    'Random Forest': build_pipeline(
        RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            class_weight=cw, random_state=42, n_jobs=-1
        )
    ),
    'Gradient Boosting': build_pipeline(
        GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8, random_state=42
        )
    ),
}

# 5. CROSS-VALIDATION
print("\n📊 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
cv_results = {}
for name, pipe in MODELS.items():
    scores = cross_val_score(pipe, X_train_raw, y_train_raw, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name[:30]}: {scores.mean():.4f} ± {scores.std():.4f}")

# 6. TRAIN & EVALUATE
print("\n🏁 Training all models...")
results = {}
for name, pipe in MODELS.items():
    print(f"  {name}...", end=' ', flush=True)
    t0 = time.time()
    pipe.fit(X_train_raw, y_train_raw)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = np.where((prec+rec)==0, 0, 2*prec*rec/(prec+rec))
    best_idx = np.argmax(f1_scores)
    opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred = (y_prob >= opt_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    cost= fp*COST_FP + fn*COST_FN
    results[name] = {
        'pipe':pipe,'y_prob':y_prob,'auc':auc,'ap':ap,
        'f1_opt':float(f1_scores[best_idx]),'f1_def':float(f1_score(y_test,(y_prob>=0.5).astype(int))),
        'precision':float(precision_score(y_test,y_pred,zero_division=0)),
        'recall':float(recall_score(y_test,y_pred)),
        'accuracy':float((y_pred==y_test).mean()),'opt_thresh':opt_thresh,'cm':cm,
        'cost':int(cost),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),
        'cv_mean':float(cv_results[name].mean()),'cv_std':float(cv_results[name].std()),
        'train_time':time.time()-t0,'prec_curve':prec,'rec_curve':rec,'thresh_curve':thresholds,
    }
    print(f"AUC={auc:.4f} AP={ap:.4f} F1={f1_scores[best_idx]:.4f} thresh={opt_thresh:.3f} ({time.time()-t0:.0f}s)")

# 7. ISOLATION FOREST
print("\n🌲 Isolation Forest...")
sc2 = StandardScaler()
X_tr_sc = sc2.fit_transform(X_train_raw)
X_te_sc = sc2.transform(X_test)
iso = IsolationForest(n_estimators=200, contamination=0.035, random_state=42, n_jobs=-1)
iso.fit(X_tr_sc)
iso_scores = -iso.score_samples(X_te_sc)
iso_pred   = (iso.predict(X_te_sc)==-1).astype(int)
iso_auc = roc_auc_score(y_test, iso_scores)
iso_f1  = f1_score(y_test, iso_pred, zero_division=0)
iso_ap  = average_precision_score(y_test, iso_scores)
results['Isolation Forest'] = {
    'pipe':None,'y_prob':iso_scores,'auc':iso_auc,'ap':iso_ap,
    'f1_opt':iso_f1,'f1_def':iso_f1,'precision':float(precision_score(y_test,iso_pred,zero_division=0)),
    'recall':float(recall_score(y_test,iso_pred,zero_division=0)),
    'accuracy':float((iso_pred==y_test).mean()),'opt_thresh':0.5,
    'cm':confusion_matrix(y_test,iso_pred),'cost':0,'tp':0,'fp':0,'fn':0,'tn':0,
    'cv_mean':0.0,'cv_std':0.0,'train_time':0,'prec_curve':None,'rec_curve':None,'thresh_curve':None,
}
print(f"  Isolation Forest: AUC={iso_auc:.4f} F1={iso_f1:.4f}")

# 8. PICK WINNER
supervised = {k:v for k,v in results.items() if k!='Isolation Forest'}
best_name  = max(supervised, key=lambda k: supervised[k]['auc'])
best = results[best_name]
MODEL = best['pipe']
print(f"\n🏆 Winner: {best_name}")
print(f"   AUC={best['auc']:.4f}  AP={best['ap']:.4f}  F1={best['f1_opt']:.4f}  thresh={best['opt_thresh']:.3f}")
print(f"   TP={best['tp']}  FP={best['fp']}  FN={best['fn']}  TN={best['tn']}")

# 9. FEATURE IMPORTANCES (works for any model)
clf_step = MODEL.named_steps['clf']
if hasattr(clf_step, 'feature_importances_'):
    raw_imp = clf_step.feature_importances_
elif hasattr(clf_step, 'coef_'):
    raw_imp = np.abs(clf_step.coef_[0])
    raw_imp = raw_imp / (raw_imp.sum() + 1e-9)
else:
    raw_imp = np.ones(len(FEATURES)) / len(FEATURES)
importances = {f: round(float(v), 6) for f, v in zip(FEATURES, raw_imp)}

# 10. SHAP
print("\n🔬 Computing SHAP values...")
SHAP_AVAILABLE = False
shap_values_dict = {}
top_shap = []
try:
    import shap
    X_test_sc_shap = MODEL.named_steps['scaler'].transform(X_test)
    X_test_df_shap = pd.DataFrame(X_test_sc_shap, columns=FEATURES)
    sample_idx = np.random.choice(len(X_test_df_shap), min(500, len(X_test_df_shap)), replace=False)
    X_shap = X_test_df_shap.iloc[sample_idx]
    explainer_shap = shap.TreeExplainer(clf_step)
    sv = explainer_shap.shap_values(X_shap)
    if isinstance(sv, list): sv = sv[1]
    mean_shap = np.abs(sv).mean(axis=0)
    shap_imp = {f: round(float(v), 6) for f, v in zip(FEATURES, mean_shap)}
    top_shap = sorted(shap_imp.items(), key=lambda x: -x[1])[:15]
    shap_values_dict = {'mean_shap': shap_imp, 'top_features': [t[0] for t in top_shap]}
    json.dump(shap_values_dict, open('models/shap_values.json','w'), indent=2)
    SHAP_AVAILABLE = True
    print(f"   Top features: {', '.join([t[0] for t in top_shap[:5]])}")
except Exception as e:
    print(f"   ⚠️  SHAP skipped: {e}")
    TRAINING_NOTES.append(f"SHAP skipped: {e}")

# 11. CHARTS
print("\n📈 Generating charts...")
MODEL_COLORS = {'Logistic Regression (Baseline)':ORANGE,'Random Forest':CYAN,
                'Gradient Boosting':PURPLE,'Isolation Forest':GREEN}

# Chart A: ROC + PR + Comparison
fig, axes = plt.subplots(1, 3, figsize=(18,5)); fig.patch.set_facecolor(DARK)
ax = axes[0]; ax.set_facecolor('#0a1020')
for nm, res in results.items():
    fpr,tpr,_ = roc_curve(y_test, res['y_prob'])
    ls = '--' if nm=='Isolation Forest' else '-'
    ax.plot(fpr, tpr, label=f"{nm[:20]} ({res['auc']:.3f})", color=MODEL_COLORS.get(nm,'#fff'), lw=2, linestyle=ls)
ax.plot([0,1],[0,1],'--',color='#333',lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves', color='white', fontsize=11)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

ax = axes[1]; ax.set_facecolor('#0a1020')
for nm, res in results.items():
    if res['prec_curve'] is not None:
        ax.plot(res['rec_curve'], res['prec_curve'],
                label=f"{nm[:20]} (AP={res['ap']:.3f})", color=MODEL_COLORS.get(nm,'#fff'), lw=2)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall Curves', color='white', fontsize=11)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_xlim([0,1]); ax.set_ylim([0,1])

ax = axes[2]; ax.set_facecolor('#0a1020')
x_ = np.arange(4); w = 0.2; metrics_ = ['auc','ap','f1_opt','recall']; mlabels = ['AUC','Avg-P','F1','Recall']
for i,(nm,res) in enumerate(results.items()):
    ax.bar(x_+i*w, [res[m] for m in metrics_], w, label=nm[:18], color=MODEL_COLORS.get(nm,'#fff'), alpha=0.85)
ax.set_xticks(x_+w*len(results)/2); ax.set_xticklabels(mlabels, fontsize=9)
ax.set_ylim([0,1.15]); ax.set_title('Model Comparison', color='white', fontsize=11)
ax.legend(fontsize=6); ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig('static/eval_curves.png', dpi=120, bbox_inches='tight', facecolor=DARK); plt.close()
print("   ✅ static/eval_curves.png")

# Chart B: Confusion matrices
all_names = list(results.keys())
fig, axes = plt.subplots(1, len(all_names), figsize=(5*len(all_names), 4)); fig.patch.set_facecolor(DARK)
if len(all_names)==1: axes=[axes]
for ax, nm in zip(axes, all_names):
    ax.set_facecolor('#0a1020')
    sns.heatmap(results[nm]['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'], cbar=False, annot_kws={'size':12})
    ax.set_title(f"{nm[:22]}\nAUC={results[nm]['auc']:.3f}", color='white', fontsize=9)
    ax.tick_params(colors='white')
plt.tight_layout(); plt.savefig('static/confusion_matrices.png', dpi=120, bbox_inches='tight', facecolor=DARK); plt.close()
print("   ✅ static/confusion_matrices.png")

# Chart C: Feature importance + SHAP
top15 = sorted(importances.items(), key=lambda x: -x[1])[:15]
fig, axes = plt.subplots(1, 2, figsize=(16,6)); fig.patch.set_facecolor(DARK)
ax = axes[0]; ax.set_facecolor('#0a1020')
n_ = [t[0] for t in top15[::-1]]; v_ = [t[1] for t in top15[::-1]]
ax.barh(n_, v_, color=[CYAN if 'V' not in x else PURPLE for x in n_], alpha=0.85)
ax.set_title(f'Top 15 Feature Importances\n{best_name}', color='white', fontsize=11)
ax.grid(True, alpha=0.2, axis='x'); ax.set_xlabel('Importance')
ax = axes[1]; ax.set_facecolor('#0a1020')
if top_shap:
    sn = [t[0] for t in top_shap[::-1]]; sv2 = [t[1] for t in top_shap[::-1]]
    ax.barh(sn, sv2, color=ORANGE, alpha=0.85)
    ax.set_title('Top 15 SHAP Mean |Values|\n(Model-Agnostic)', color='white', fontsize=11)
    ax.grid(True, alpha=0.2, axis='x'); ax.set_xlabel('Mean |SHAP|')
else:
    ax.text(0.5, 0.5, 'Install shap:\npip install shap\nThen re-run train_model.py',
            ha='center', va='center', color=ORANGE, fontsize=13, transform=ax.transAxes)
    ax.set_title('SHAP Explainability', color='white')
plt.tight_layout(); plt.savefig('static/feature_importance.png', dpi=120, bbox_inches='tight', facecolor=DARK); plt.close()
print("   ✅ static/feature_importance.png")

# Chart D: Cost + threshold
fig, axes = plt.subplots(1, 2, figsize=(14,5)); fig.patch.set_facecolor(DARK)
ax = axes[0]; ax.set_facecolor('#0a1020')
sup_r = {k:v for k,v in results.items() if k!='Isolation Forest'}
mnames = list(sup_r.keys()); costs_ = [sup_r[k]['cost'] for k in mnames]
ax.bar(range(len(mnames)), costs_, color=[MODEL_COLORS.get(k,'#fff') for k in mnames], alpha=0.85)
ax.set_xticks(range(len(mnames))); ax.set_xticklabels([n[:20] for n in mnames], rotation=12, fontsize=8)
ax.set_title(f'Cost Analysis (FN={COST_FN}x, FP={COST_FP}x)', color='white', fontsize=11)
ax.set_ylabel('Total Cost (lower=better)'); ax.grid(True, alpha=0.2, axis='y')
for i,c in enumerate(costs_): ax.text(i, c+max(costs_)*0.02 if max(costs_)>0 else 0.01, str(c), ha='center', color='white', fontsize=10)
ax = axes[1]; ax.set_facecolor('#0a1020')
if best['prec_curve'] is not None:
    pc,rc,tc = best['prec_curve'],best['rec_curve'],best['thresh_curve']
    f1t = np.where((pc[:-1]+rc[:-1])==0,0,2*pc[:-1]*rc[:-1]/(pc[:-1]+rc[:-1]))
    ax.plot(tc,f1t,color=CYAN,lw=2,label='F1')
    ax.plot(tc,pc[:-1],color=GREEN,lw=2,label='Precision')
    ax.plot(tc,rc[:-1],color=ORANGE,lw=2,label='Recall')
    ax.axvline(best['opt_thresh'],color=RED,lw=2,linestyle='--',label=f"Opt={best['opt_thresh']:.3f}")
    ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
    ax.set_title(f'Threshold Tuning — {best_name[:25]}', color='white', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xlim([0,1])
plt.tight_layout(); plt.savefig('static/cost_threshold.png', dpi=120, bbox_inches='tight', facecolor=DARK); plt.close()
print("   ✅ static/cost_threshold.png")

# Chart E: CV box plot
fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(DARK); ax.set_facecolor('#0a1020')
cv_data = [cv_results[k] for k in MODELS]
bp = ax.boxplot(cv_data, patch_artist=True, medianprops={'color':RED,'linewidth':2})
for patch,nm in zip(bp['boxes'], MODELS): patch.set_facecolor(MODEL_COLORS.get(nm,'#fff')); patch.set_alpha(0.6)
ax.set_xticklabels([k[:22] for k in MODELS], rotation=10, fontsize=9)
ax.set_ylabel('ROC-AUC'); ax.set_title(f'{CV_FOLDS}-Fold Cross-Validation', color='white', fontsize=12)
ax.grid(True, alpha=0.2, axis='y'); ax.set_ylim([0.5,1.05])
plt.tight_layout(); plt.savefig('static/cross_validation.png', dpi=120, bbox_inches='tight', facecolor=DARK); plt.close()
print("   ✅ static/cross_validation.png")

# 12. RAG STORE (uses already-engineered X — no V-column mismatch)
print("\n🗄️  Building RAG store...")
fraud_index = y[y==1].index
X_ff = X.loc[fraud_index]; df_ff = df.loc[fraud_index]
if len(X_ff) > MAX_RAG:
    si = X_ff.sample(n=MAX_RAG, random_state=42).index
    X_fraud = X_ff.loc[si].reset_index(drop=True); df_fraud = df_ff.loc[si].reset_index(drop=True)
else:
    X_fraud = X_ff.reset_index(drop=True); df_fraud = df_ff.reset_index(drop=True)
rag_vecs = X_fraud[FEATURES].fillna(-999).values.astype(np.float32)
norms = np.linalg.norm(rag_vecs, axis=1, keepdims=True); norms[norms==0]=1
rag_normed = rag_vecs / norms
rag_meta = [{'case_id':f"TXN-{int(df_fraud.iloc[i].get('TransactionID',i))}",
             'amount':round(float(df_fraud.iloc[i]['TransactionAmt']),2),
             'product':str(df_fraud.iloc[i].get('ProductCD','W')),
             'hour':int((float(df_fraud.iloc[i].get('TransactionDT',0))//3600)%24),
             'card_type':str(df_fraud.iloc[i].get('card4','visa')),
             'email':str(df_fraud.iloc[i].get('P_emaildomain','unknown')),
             'outcome':'Fraud Confirmed'} for i in range(len(df_fraud))]
print(f"   RAG cases: {len(rag_meta):,}")

# 13. SAVE
print("\n💾 Saving all artifacts...")
comparison_table = [{'model':nm,'auc':round(r['auc'],4),'ap':round(r['ap'],4),
    'f1':round(r['f1_opt'],4),'precision':round(r['precision'],4),'recall':round(r['recall'],4),
    'accuracy':round(r['accuracy'],4),'cv_mean':round(r['cv_mean'],4),'cv_std':round(r['cv_std'],4),
    'cost':r['cost'],'threshold':round(r['opt_thresh'],4),'is_winner':nm==best_name}
    for nm,r in results.items()]

METRICS = {
    'model_name':best_name,'accuracy':round(best['accuracy'],4),'roc_auc':round(best['auc'],4),
    'avg_precision':round(best['ap'],4),'f1':round(best['f1_opt'],4),'f1_default':round(best['f1_def'],4),
    'precision':round(best['precision'],4),'recall':round(best['recall'],4),
    'opt_threshold':round(best['opt_thresh'],4),'cv_mean':round(best['cv_mean'],4),
    'cv_std':round(best['cv_std'],4),'tp':best['tp'],'fp':best['fp'],'fn':best['fn'],'tn':best['tn'],
    'cost':best['cost'],'cost_fn':COST_FN,'cost_fp':COST_FP,'smote_used':SMOTE_USED,
    'shap_available':SHAP_AVAILABLE,
    'n_train_raw': int(len(X_train_raw)),
    'n_train_resampled': int(len(X_train_resampled)),
    'n_test':int(len(X_test)),
    'n_features':len(FEATURES),
    'cv_folds': CV_FOLDS,
    'feature_importances':importances,
    'comparison_table':comparison_table,
    'training_notes': TRAINING_NOTES,
}

pickle.dump(MODEL, open('models/fraud_model.pkl','wb'))
json.dump(METRICS,   open('models/metrics.json','w'), indent=2)
json.dump(FEATURES,  open('models/features.json','w'), indent=2)
json.dump(CATEGORY_MAPS, open('models/category_maps.json','w'), indent=2)
json.dump(rag_meta,  open('models/rag_meta.json','w'), indent=2)
json.dump({'threshold':best['opt_thresh']}, open('models/threshold.json','w'), indent=2)
np.save('models/rag_vectors.npy', rag_normed)

print("\n"+"="*60)
print(f"  ✅ DONE!  Best: {best_name}")
print(f"  ROC-AUC={best['auc']:.4f}  F1={best['f1_opt']:.4f}  Threshold={best['opt_thresh']:.3f}")
print(f"  SMOTE={'✅' if SMOTE_USED else '❌'}  SHAP={'✅' if SHAP_AVAILABLE else '❌'}")
print("="*60)
print("\n🚀 Now run:  python app.py")
