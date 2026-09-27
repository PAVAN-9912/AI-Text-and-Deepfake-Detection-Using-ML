import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models", "text_final")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("PHASE B — ERROR ANALYSIS ON HELD-OUT VALIDATION SET (N=998)")
print("=" * 80)

# Import extract_features_single and get_length_group from predict_text_final
from predict_text_final import (
    word_vectorizer,
    char_vectorizer,
    multi_char_vectorizer,
    linguistic_scaler,
    classifier,
    calibrator,
    config,
    transformer_model,
    length_thresholds,
    alpha_short_decay,
    extract_features_single,
    get_length_group
)

# Load the verified validation set
validation_csv = os.path.join(MODEL_DIR, "validation_results.csv")
if not os.path.exists(validation_csv):
    raise FileNotFoundError(f"Missing validation_results.csv in {MODEL_DIR}")

val_df = pd.read_csv(validation_csv)
print(f"Loaded validation set: {len(val_df)} rows")
print(f"Distribution: Human (0) = {(val_df['label'] == 0).sum()}, AI (1) = {(val_df['label'] == 1).sum()}")

# Run inference over all validation samples
predictions = []
ai_probabilities = []
human_probabilities = []
confidences = []
applied_thresholds = []
length_groups = []

for idx, row in val_df.iterrows():
    text = str(row["text"]).strip()
    lg = get_length_group(text)
    thresh = length_thresholds.get(lg, 0.50)
    
    # Features
    X_w = word_vectorizer.transform([text])
    X_c = char_vectorizer.transform([text])
    X_mc = multi_char_vectorizer.transform([text])
    emb = transformer_model.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    raw_scalar, wc, code_lik, is_ger = extract_features_single(text)
    X_s = linguistic_scaler.transform(raw_scalar)
    decay = np.array([[alpha_short_decay * np.exp(-wc / 15.0)]], dtype=np.float32)
    X_si = csr_matrix((emb * decay).astype(np.float32))
    
    X_full = hstack([X_w, X_c, X_mc, csr_matrix(emb), X_si, csr_matrix(X_s)], format="csr")
    
    score = classifier.decision_function(X_full).reshape(-1, 1)
    probs = calibrator.predict_proba(score)[0]
    p_ai = float(probs[1])
    p_hum = float(probs[0])
    
    is_ai = p_ai >= thresh
    pred = 1 if is_ai else 0
    conf = p_ai if is_ai else p_hum
    
    predictions.append(pred)
    ai_probabilities.append(p_ai)
    human_probabilities.append(p_hum)
    confidences.append(conf)
    applied_thresholds.append(thresh)
    length_groups.append(lg)

val_df["eval_prediction"] = predictions
val_df["eval_ai_prob"] = ai_probabilities
val_df["eval_human_prob"] = human_probabilities
val_df["eval_confidence"] = confidences
val_df["eval_threshold"] = applied_thresholds
val_df["eval_length_group"] = length_groups

y_true = val_df["label"].values
y_pred = np.array(predictions)
y_prob = np.array(ai_probabilities)

# Overall Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nOverall Validation Results:")
print(f"  Accuracy : {acc*100:.2f}% ({tn+tp}/{len(y_true)})")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall   : {rec*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print(f"  ROC-AUC  : {auc*100:.2f}%")
print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Classify error types
val_df["error_category"] = "Correct"
val_df.loc[(val_df["label"] == 0) & (val_df["eval_prediction"] == 1), "error_category"] = "False Positive"
val_df.loc[(val_df["label"] == 1) & (val_df["eval_prediction"] == 0), "error_category"] = "False Negative"

# Analysis by length tiers
length_tiers = ["very_short", "short", "medium", "long"]
tier_reports = {}

for tier in length_tiers:
    sub = val_df[val_df["eval_length_group"] == tier]
    sub_y = sub["label"].values
    sub_pred = sub["eval_prediction"].values
    sub_prob = sub["eval_ai_prob"].values
    
    sub_acc = accuracy_score(sub_y, sub_pred) if len(sub) > 0 else 0.0
    sub_prec = precision_score(sub_y, sub_pred, zero_division=0) if len(sub) > 0 else 0.0
    sub_rec = recall_score(sub_y, sub_pred, zero_division=0) if len(sub) > 0 else 0.0
    sub_f1 = f1_score(sub_y, sub_pred, zero_division=0) if len(sub) > 0 else 0.0
    try:
        sub_auc = roc_auc_score(sub_y, sub_prob) if len(np.unique(sub_y)) > 1 else 0.0
    except Exception:
        sub_auc = 0.0
        
    sub_fp = int(((sub_y == 0) & (sub_pred == 1)).sum())
    sub_fn = int(((sub_y == 1) & (sub_pred == 0)).sum())
    sub_tn = int(((sub_y == 0) & (sub_pred == 0)).sum())
    sub_tp = int(((sub_y == 1) & (sub_pred == 1)).sum())
    
    tier_reports[tier] = {
        "count": len(sub),
        "threshold": float(length_thresholds.get(tier, 0.50)),
        "accuracy": float(round(sub_acc, 4)),
        "precision": float(round(sub_prec, 4)),
        "recall": float(round(sub_rec, 4)),
        "f1": float(round(sub_f1, 4)),
        "roc_auc": float(round(sub_auc, 4)),
        "mean_predicted_ai_prob": float(round(np.mean(sub_prob), 4)) if len(sub_prob) > 0 else 0.0,
        "prob_percentiles": {
            "p25": float(round(np.percentile(sub_prob, 25), 4)) if len(sub_prob) > 0 else 0.0,
            "median": float(round(np.median(sub_prob), 4)) if len(sub_prob) > 0 else 0.0,
            "p75": float(round(np.percentile(sub_prob, 75), 4)) if len(sub_prob) > 0 else 0.0
        },
        "true_negatives": sub_tn,
        "false_positives": sub_fp,
        "false_negatives": sub_fn,
        "true_positives": sub_tp
    }

print("\nPerformance by Length Tier:")
print("-" * 75)
for tier, data in tier_reports.items():
    print(f"Tier: {tier:12s} | N={data['count']:3d} | Acc={data['accuracy']*100:6.2f}% | F1={data['f1']*100:6.2f}% | AUC={data['roc_auc']*100:6.2f}% | FP={data['false_positives']:2d} | FN={data['false_negatives']:2d} | Mean AI Prob={data['mean_predicted_ai_prob']*100:5.2f}%")

# Analysis of False Positives
fps = val_df[val_df["error_category"] == "False Positive"]
fns = val_df[val_df["error_category"] == "False Negative"]

fp_summary = {
    "count": len(fps),
    "by_length": fps["eval_length_group"].value_counts().to_dict(),
    "by_dataset": fps["dataset"].value_counts().to_dict() if "dataset" in fps.columns else {},
    "mean_predicted_ai_prob": float(round(fps["eval_ai_prob"].mean(), 4)) if len(fps) > 0 else 0.0,
    "prob_percentiles": {
        "p25": float(round(np.percentile(fps["eval_ai_prob"], 25), 4)) if len(fps) > 0 else 0.0,
        "median": float(round(np.median(fps["eval_ai_prob"]), 4)) if len(fps) > 0 else 0.0,
        "p75": float(round(np.percentile(fps["eval_ai_prob"], 75), 4)) if len(fps) > 0 else 0.0
    }
}

fn_summary = {
    "count": len(fns),
    "by_length": fns["eval_length_group"].value_counts().to_dict(),
    "by_dataset": fns["dataset"].value_counts().to_dict() if "dataset" in fns.columns else {},
    "mean_predicted_ai_prob": float(round(fns["eval_ai_prob"].mean(), 4)) if len(fns) > 0 else 0.0,
    "prob_percentiles": {
        "p25": float(round(np.percentile(fns["eval_ai_prob"], 25), 4)) if len(fns) > 0 else 0.0,
        "median": float(round(np.median(fns["eval_ai_prob"]), 4)) if len(fns) > 0 else 0.0,
        "p75": float(round(np.percentile(fns["eval_ai_prob"], 75), 4)) if len(fns) > 0 else 0.0
    }
}

full_report = {
    "overall": {
        "samples": len(val_df),
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1": float(round(f1, 4)),
        "roc_auc": float(round(auc, 4)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    },
    "by_length_tier": tier_reports,
    "false_positives": fp_summary,
    "false_negatives": fn_summary
}

# Save report JSON
json_path = os.path.join(REPORTS_DIR, "text_error_analysis.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_report, f, indent=2)
print(f"\nSaved Error Analysis JSON: {json_path}")

# Save errors CSV
errors_df = val_df[val_df["error_category"] != "Correct"].copy()
csv_path = os.path.join(REPORTS_DIR, "text_error_analysis.csv")
errors_df.to_csv(csv_path, index=False)
print(f"Saved Error Analysis CSV : {csv_path} ({len(errors_df)} error records)")

# Print representative FP and FN examples
print("\nRepresentative False Positive Examples (Human classified as AI):")
print("-" * 75)
for i, (_, r) in enumerate(fps.head(4).iterrows()):
    print(f"[{i+1}] Tier: {r['eval_length_group']:10s} | Dataset: {r.get('dataset', 'N/A')} | AI Prob: {r['eval_ai_prob']*100:.1f}% | Thresh: {r['eval_threshold']:.2f}")
    print(f"    Text: {repr(str(r['text'])[:110])}...\n")

print("Representative False Negative Examples (AI classified as Human):")
print("-" * 75)
for i, (_, r) in enumerate(fns.head(4).iterrows()):
    print(f"[{i+1}] Tier: {r['eval_length_group']:10s} | Dataset: {r.get('dataset', 'N/A')} | AI Prob: {r['eval_ai_prob']*100:.1f}% | Thresh: {r['eval_threshold']:.2f}")
    print(f"    Text: {repr(str(r['text'])[:110])}...\n")

print("=" * 80)
print("PHASE B — ERROR ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
