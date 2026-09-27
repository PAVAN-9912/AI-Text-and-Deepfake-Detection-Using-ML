import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    total_samples = len(y_true)
    
    bin_stats = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            actual_rate = np.mean(y_true[mask])
            mean_prob = np.mean(y_prob[mask])
            abs_diff = np.abs(actual_rate - mean_prob)
            ece += (count / total_samples) * abs_diff
            bin_stats.append({
                "range": f"{int(bins[i]*100)}–{int(bins[i+1]*100)}%",
                "samples": count,
                "mean_pred": round(mean_prob * 100, 2),
                "actual_ai": round(actual_rate * 100, 2),
                "abs_diff": round(abs_diff * 100, 2)
            })
        else:
            bin_stats.append({
                "range": f"{int(bins[i]*100)}–{int(bins[i+1]*100)}%",
                "samples": 0,
                "mean_pred": 0.0,
                "actual_ai": 0.0,
                "abs_diff": 0.0
            })
    return ece, bin_stats

def main():
    print("=" * 80)
    print("PROBABILITY CALIBRATION EXPERIMENT (HELD-OUT VALIDATION SET, N = 1,089)")
    print("=" * 80)

    val_csv_path = "models/text_final_v2_improved/validation_results.csv"
    df_val = pd.read_csv(val_csv_path)
    
    print(f"Loaded {len(df_val)} samples from {val_csv_path}")
    print(f"Ground-truth column : 'label' (Human = 0: {(df_val['label']==0).sum()}, AI = 1: {(df_val['label']==1).sum()})")
    print(f"Model prob column   : 'probability'")
    print(f"Model pred column   : 'prediction'\n")

    y_true = df_val["label"].values
    p_uncal = df_val["probability"].values
    
    # -------------------------------------------------------------
    # 1. OUT-OF-FOLD CALIBRATION (Avoid calibration leakage)
    # -------------------------------------------------------------
    # Using 10-Fold Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    p_sigmoid_oof = np.zeros_like(p_uncal)
    p_isotonic_oof = np.zeros_like(p_uncal)

    # Convert probabilities to log-odds (logits) with clipping for numerical stability
    eps = 1e-7
    p_clipped = np.clip(p_uncal, eps, 1.0 - eps)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    for train_idx, val_idx in skf.split(logits, y_true):
        # A. Sigmoid / Platt Scaling (Logistic Regression on Logits)
        platt_scaler = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        platt_scaler.fit(logits[train_idx], y_true[train_idx])
        p_sigmoid_oof[val_idx] = platt_scaler.predict_proba(logits[val_idx])[:, 1]
        
        # B. Isotonic Regression (Non-parametric isotonic mapping on probabilities)
        iso_scaler = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_scaler.fit(p_uncal[train_idx], y_true[train_idx])
        p_isotonic_oof[val_idx] = iso_scaler.predict(p_uncal[val_idx])

    # -------------------------------------------------------------
    # 2. EVALUATE METRICS
    # -------------------------------------------------------------
    methods = {
        "Uncalibrated (Baseline)": p_uncal,
        "Sigmoid / Platt (OOF)": p_sigmoid_oof,
        "Isotonic (OOF)": p_isotonic_oof
    }

    results = []
    bin_tables = {}

    for name, probs in methods.items():
        # Calibration quality metrics
        brier = brier_score_loss(y_true, probs)
        logloss = log_loss(y_true, np.clip(probs, 1e-15, 1.0 - 1e-15))
        auc = roc_auc_score(y_true, probs)
        ece, b_stats = compute_ece(y_true, probs, n_bins=10)
        bin_tables[name] = b_stats
        
        # Classification performance at decision threshold = 0.41
        pred_041 = (probs >= 0.41).astype(int)
        acc_041 = accuracy_score(y_true, pred_041)
        prec_041 = precision_score(y_true, pred_041, zero_division=0)
        rec_041 = recall_score(y_true, pred_041, zero_division=0)
        f1_041 = f1_score(y_true, pred_041, zero_division=0)
        cm_041 = confusion_matrix(y_true, pred_041)
        fp_041 = cm_041[0, 1]
        fn_041 = cm_041[1, 0]

        # Classification performance at standard threshold = 0.50
        pred_050 = (probs >= 0.50).astype(int)
        acc_050 = accuracy_score(y_true, pred_050)
        f1_050 = f1_score(y_true, pred_050, zero_division=0)
        
        results.append({
            "name": name,
            "brier": brier,
            "log_loss": logloss,
            "roc_auc": auc,
            "ece": ece,
            "acc_041": acc_041,
            "prec_041": prec_041,
            "rec_041": rec_041,
            "f1_041": f1_041,
            "fp_041": fp_041,
            "fn_041": fn_041,
            "acc_050": acc_050,
            "f1_050": f1_050
        })

    # Print Comparison Table
    print("=" * 80)
    print("1. PROBABILITY CALIBRATION & ERROR COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Uncalibrated':<16} | {'Sigmoid (Platt)':<16} | {'Isotonic':<16}")
    print("-" * 80)
    print(f"{'Brier Score (Lower=Better)':<25} | {results[0]['brier']:.4f}           | {results[1]['brier']:.4f}           | {results[2]['brier']:.4f}")
    print(f"{'Log Loss (Lower=Better)':<25} | {results[0]['log_loss']:.4f}           | {results[1]['log_loss']:.4f}           | {results[2]['log_loss']:.4f}")
    print(f"{'ROC-AUC (Higher=Better)':<25} | {results[0]['roc_auc']*100:.2f}%          | {results[1]['roc_auc']*100:.2f}%          | {results[2]['roc_auc']*100:.2f}%")
    print(f"{'ECE (Lower=Better)':<25} | {results[0]['ece']*100:.2f}%          | {results[1]['ece']*100:.2f}%          | {results[2]['ece']*100:.2f}%")
    print("-" * 80)
    print(f"{'Accuracy (@ 0.41 Thresh)':<25} | {results[0]['acc_041']*100:.2f}%          | {results[1]['acc_041']*100:.2f}%          | {results[2]['acc_041']*100:.2f}%")
    print(f"{'Precision (@ 0.41 Thresh)':<25} | {results[0]['prec_041']*100:.2f}%          | {results[1]['prec_041']*100:.2f}%          | {results[2]['prec_041']*100:.2f}%")
    print(f"{'Recall (@ 0.41 Thresh)':<25} | {results[0]['rec_041']*100:.2f}%          | {results[1]['rec_041']*100:.2f}%          | {results[2]['rec_041']*100:.2f}%")
    print(f"{'F1-Score (@ 0.41 Thresh)':<25} | {results[0]['f1_041']*100:.2f}%          | {results[1]['f1_041']*100:.2f}%          | {results[2]['f1_041']*100:.2f}%")
    print(f"{'False Positives (FP)':<25} | {results[0]['fp_041']:<16} | {results[1]['fp_041']:<16} | {results[2]['fp_041']:<16}")
    print(f"{'False Negatives (FN)':<25} | {results[0]['fn_041']:<16} | {results[1]['fn_041']:<16} | {results[2]['fn_041']:<16}")

    # -------------------------------------------------------------
    # 3. CALIBRATION BIN TABLES
    # -------------------------------------------------------------
    for name in methods.keys():
        print("\n" + "=" * 80)
        print(f"CALIBRATION BIN TABLE: {name.upper()}")
        print("=" * 80)
        print(f"{'Probability Range':<20} | {'Samples':<10} | {'Mean Pred Prob':<18} | {'Actual AI Rate':<16} | {'Gap (|Pred-Actual|)':<20}")
        print("-" * 90)
        for row in bin_tables[name]:
            print(f"{row['range']:<20} | {row['samples']:<10} | {row['mean_pred']:>6.2f}%            | {row['actual_ai']:>6.2f}%          | {row['abs_diff']:>6.2f}%")

if __name__ == "__main__":
    main()
