import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Import Flask app from app.py
from app import app as flask_app

def main():
    print("=" * 80)
    print("PHASE 1.5 TEXT INFERENCE PIPELINE VALIDATION EVALUATION (1,089 SAMPLES)")
    print("=" * 80)

    # 1. Load exact 1,089 validation samples
    val_csv_path = "models/text_final_v2_improved/validation_results.csv"
    val_df = pd.read_csv(val_csv_path)
    print(f"Loaded {len(val_df)} validation samples from {val_csv_path}")
    print(f"Class distribution: {val_df['label'].value_counts().to_dict()}\n")

    client = flask_app.test_client()
    records = []
    
    total = len(val_df)
    for idx, row in val_df.iterrows():
        if (idx + 1) % 150 == 0 or idx == 0 or idx == total - 1:
            print(f"Processing sample {idx+1}/{total}...")
            
        text = str(row["text"])
        true_label = int(row["label"])
        orig_prob = float(row.get("probability", np.nan))
        orig_pred = int(row.get("prediction", 1 if orig_prob >= 0.41 else 0))
        
        # Run CURRENT Phase 1 text inference pipeline via Flask test client
        response = client.post("/api/analyze/text", json={"text": text})
        res = response.get_json()
        
        word_count = res.get("word_count", len(text.split()))
        length_group = res.get("length_group", "unknown")
        result_str = res.get("result")
        conf = res.get("confidence")
        
        # Phase 1 prediction mapping (1 for AI, 0 for Human / Uncertain / Insufficient)
        if result_str == "AI-GENERATED":
            phase1_pred = 1
        elif result_str == "HUMAN-WRITTEN":
            phase1_pred = 0
        else: # UNCERTAIN or INSUFFICIENT_EVIDENCE
            phase1_pred = -1
            
        global_prob = res.get("global_ai_probability", res.get("ai_probability"))
        if global_prob is not None:
            global_prob = global_prob / 100.0
            
        mean_chunk_prob = res.get("mean_chunk_ai_probability")
        if mean_chunk_prob is not None:
            mean_chunk_prob = mean_chunk_prob / 100.0
            
        peak_chunk_prob = res.get("peak_chunk_ai_probability")
        if peak_chunk_prob is not None:
            peak_chunk_prob = peak_chunk_prob / 100.0
            
        ai_chunk_ratio = res.get("ai_chunk_ratio")
        if ai_chunk_ratio is not None:
            ai_chunk_ratio = ai_chunk_ratio / 100.0
            
        num_chunks = res.get("number_of_chunks", len(res.get("chunks", [])))
        
        records.append({
            "idx": idx,
            "text": text,
            "word_count": word_count,
            "length_group": length_group,
            "true_label": true_label,
            "orig_pred": orig_pred,
            "orig_prob": orig_prob,
            "phase1_result_str": result_str,
            "phase1_pred": phase1_pred,
            "phase1_conf": conf,
            "global_ai_prob": global_prob,
            "mean_chunk_ai_prob": mean_chunk_prob,
            "peak_chunk_ai_prob": peak_chunk_prob,
            "ai_chunk_ratio": ai_chunk_ratio,
            "num_chunks": num_chunks
        })

    df_eval = pd.DataFrame(records)

    # -------------------------------------------------------------
    # METRIC CALCULATIONS
    # -------------------------------------------------------------
    y_true = df_eval["true_label"].values
    
    uncertain_count = (df_eval["phase1_pred"] == -1).sum()
    print(f"\nCompleted inference. Uncertain / Insufficient evidence samples: {uncertain_count}")
    
    # Binary predictions (treating non-AI as 0)
    y_pred_phase1 = np.where(df_eval["phase1_pred"] == 1, 1, 0)
    y_prob_phase1 = df_eval["global_ai_prob"].fillna(0.0).values
    
    # Original baseline metrics
    y_pred_orig = df_eval["orig_pred"].values
    y_prob_orig = df_eval["orig_prob"].values

    acc_orig = accuracy_score(y_true, y_pred_orig)
    prec_orig = precision_score(y_true, y_pred_orig, zero_division=0)
    rec_orig = recall_score(y_true, y_pred_orig, zero_division=0)
    f1_orig = f1_score(y_true, y_pred_orig, zero_division=0)
    auc_orig = roc_auc_score(y_true, y_prob_orig)
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    tn_orig, fp_orig, fn_orig, tp_orig = cm_orig.ravel()

    acc_p1 = accuracy_score(y_true, y_pred_phase1)
    prec_p1 = precision_score(y_true, y_pred_phase1, zero_division=0)
    rec_p1 = recall_score(y_true, y_pred_phase1, zero_division=0)
    f1_p1 = f1_score(y_true, y_pred_phase1, zero_division=0)
    auc_p1 = roc_auc_score(y_true, y_prob_phase1)
    cm_p1 = confusion_matrix(y_true, y_pred_phase1)
    tn_p1, fp_p1, fn_p1, tp_p1 = cm_p1.ravel()

    print("\n" + "=" * 80)
    print("1. OVERALL METRIC COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'Phase 1':<15} | {'Difference':<15}")
    print("-" * 72)
    print(f"{'Accuracy':<20} | {acc_orig*100:6.2f}%         | {acc_p1*100:6.2f}%         | {(acc_p1-acc_orig)*100:+6.2f}%")
    print(f"{'Precision':<20} | {prec_orig*100:6.2f}%         | {prec_p1*100:6.2f}%         | {(prec_p1-prec_orig)*100:+6.2f}%")
    print(f"{'Recall':<20} | {rec_orig*100:6.2f}%         | {rec_p1*100:6.2f}%         | {(rec_p1-rec_orig)*100:+6.2f}%")
    print(f"{'F1-Score':<20} | {f1_orig*100:6.2f}%         | {f1_p1*100:6.2f}%         | {(f1_p1-f1_orig)*100:+6.2f}%")
    print(f"{'ROC-AUC':<20} | {auc_orig*100:6.2f}%         | {auc_p1*100:6.2f}%         | {(auc_p1-auc_orig)*100:+6.2f}%")
    print(f"{'False Positives (FP)':<20} | {fp_orig:<15} | {fp_p1:<15} | {fp_p1-fp_orig:+d}")
    print(f"{'False Negatives (FN)':<20} | {fn_orig:<15} | {fn_p1:<15} | {fn_p1-fn_orig:+d}")
    print(f"{'True Positives (TP)':<20} | {tp_orig:<15} | {tp_p1:<15} | {tp_p1-tp_orig:+d}")
    print(f"{'True Negatives (TN)':<20} | {tn_orig:<15} | {tn_p1:<15} | {tn_p1-tn_orig:+d}")

    # -------------------------------------------------------------
    # BREAKDOWN BY LENGTH GROUP
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. METRICS BY LENGTH GROUP")
    print("=" * 80)
    for lg in ["very_short", "short", "medium", "long"]:
        sub = df_eval[df_eval["length_group"] == lg]
        if len(sub) == 0:
            continue
        sub_y_true = sub["true_label"].values
        sub_y_orig = sub["orig_pred"].values
        sub_y_p1 = np.where(sub["phase1_pred"] == 1, 1, 0)
        
        cm_s_orig = confusion_matrix(sub_y_true, sub_y_orig, labels=[0, 1])
        cm_s_p1 = confusion_matrix(sub_y_true, sub_y_p1, labels=[0, 1])
        
        print(f"\n--- Group: {lg.upper()} (N = {len(sub)} | Human: {(sub_y_true==0).sum()}, AI: {(sub_y_true==1).sum()}) ---")
        print(f"  Baseline: Accuracy={accuracy_score(sub_y_true, sub_y_orig)*100:.2f}% | F1={f1_score(sub_y_true, sub_y_orig, zero_division=0)*100:.2f}% | FP={cm_s_orig[0,1]} | FN={cm_s_orig[1,0]}")
        print(f"  Phase 1 : Accuracy={accuracy_score(sub_y_true, sub_y_p1)*100:.2f}% | F1={f1_score(sub_y_true, sub_y_p1, zero_division=0)*100:.2f}% | FP={cm_s_p1[0,1]} | FN={cm_s_p1[1,0]}")

    # -------------------------------------------------------------
    # INVESTIGATION QUESTIONS A, B, C
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. INVESTIGATION ANALYSIS (QUESTIONS A, B, C)")
    print("=" * 80)

    # B. How many validation samples actually receive chunk analysis? (num_chunks > 1)
    chunked_samples = df_eval[df_eval["num_chunks"] > 1]
    print(f"B. Validation samples receiving multi-chunk analysis (>1 chunk): {len(chunked_samples)} / {len(df_eval)} ({len(chunked_samples)/len(df_eval)*100:.2f}%)")
    print(f"   Single-chunk samples: {len(df_eval) - len(chunked_samples)}")

    # C1. Mean chunk prob substantially different from global prob (|mean - global| >= 0.15) on chunked samples
    diff_mask = (chunked_samples["mean_chunk_ai_prob"] - chunked_samples["global_ai_prob"]).abs() >= 0.15
    print(f"C1. Chunked samples with |mean_chunk - global| >= 15%: {diff_mask.sum()} / {len(chunked_samples)} ({diff_mask.sum()/len(chunked_samples)*100:.2f}%)")

    # C2. High peak prob (>= 0.75) but low global prob (< 0.41)
    peak_high_global_low = df_eval[(df_eval["peak_chunk_ai_prob"] >= 0.75) & (df_eval["global_ai_prob"] < 0.41)]
    print(f"C2. Samples with Peak >= 75% but Global < 41%: {len(peak_high_global_low)} / {len(df_eval)} ({len(peak_high_global_low)/len(df_eval)*100:.2f}%)")
    if len(peak_high_global_low) > 0:
        print(f"    Human: {(peak_high_global_low['true_label']==0).sum()} | AI: {(peak_high_global_low['true_label']==1).sum()}")

    # C3. High AI chunk ratio (>= 0.50) but Human ground truth
    high_ratio_human = df_eval[(df_eval["ai_chunk_ratio"] >= 0.50) & (df_eval["true_label"] == 0) & (df_eval["num_chunks"] > 1)]
    print(f"C3. Human samples with AI chunk ratio >= 50%: {len(high_ratio_human)} / {(df_eval['true_label']==0).sum()} ({len(high_ratio_human)/(df_eval['true_label']==0).sum()*100:.2f}%)")

    # -------------------------------------------------------------
    # D. 20 STRONGEST FALSE POSITIVES & 20 STRONGEST FALSE NEGATIVES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. TOP 20 STRONGEST FALSE POSITIVES (Human true label = 0, Predicted = 1)")
    print("=" * 80)
    fps = df_eval[(df_eval["true_label"] == 0) & (df_eval["phase1_pred"] == 1)].copy()
    fps = fps.sort_values(by="global_ai_prob", ascending=False).head(20)
    
    for i, (_, r) in enumerate(fps.iterrows(), 1):
        g_prob = f"{r['global_ai_prob']*100:.2f}%" if pd.notnull(r['global_ai_prob']) else "N/A"
        m_prob = f"{r['mean_chunk_ai_prob']*100:.2f}%" if pd.notnull(r['mean_chunk_ai_prob']) else "N/A"
        p_prob = f"{r['peak_chunk_ai_prob']*100:.2f}%" if pd.notnull(r['peak_chunk_ai_prob']) else "N/A"
        ratio = f"{r['ai_chunk_ratio']*100:.1f}%" if pd.notnull(r['ai_chunk_ratio']) else "N/A"
        print(f"{i:2d}. [Words: {r['word_count']:3d} | {r['length_group']}] Global AI: {g_prob} | Mean: {m_prob} | Peak: {p_prob} | Ratio: {ratio}")
        print(f"    Text: {r['text'][:120]}...\n")

    print("\n" + "=" * 80)
    print("5. TOP 20 STRONGEST FALSE NEGATIVES (AI true label = 1, Predicted = 0)")
    print("=" * 80)
    fns = df_eval[(df_eval["true_label"] == 1) & (df_eval["phase1_pred"] == 0)].copy()
    fns = fns.sort_values(by="global_ai_prob", ascending=True).head(20)
    
    for i, (_, r) in enumerate(fns.iterrows(), 1):
        g_prob = f"{r['global_ai_prob']*100:.2f}%" if pd.notnull(r['global_ai_prob']) else "N/A"
        m_prob = f"{r['mean_chunk_ai_prob']*100:.2f}%" if pd.notnull(r['mean_chunk_ai_prob']) else "N/A"
        p_prob = f"{r['peak_chunk_ai_prob']*100:.2f}%" if pd.notnull(r['peak_chunk_ai_prob']) else "N/A"
        ratio = f"{r['ai_chunk_ratio']*100:.1f}%" if pd.notnull(r['ai_chunk_ratio']) else "N/A"
        print(f"{i:2d}. [Words: {r['word_count']:3d} | {r['length_group']}] Global AI: {g_prob} | Mean: {m_prob} | Peak: {p_prob} | Ratio: {ratio}")
        print(f"    Text: {r['text'][:120]}...\n")

if __name__ == "__main__":
    main()
