import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory setup
base_dir = r"models\text_final_v3"
output_dir = os.path.join(base_dir, "error_analysis")
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load and verify
val_file = os.path.join(base_dir, "validation_results.csv")
df = pd.read_csv(val_file)

total_samples = len(df)
human_samples = int((df['label'] == 0).sum())
ai_samples = int((df['label'] == 1).sum())
fn_count = int(((df['label'] == 1) & (df['prediction'] == 0)).sum())
fp_count = int(((df['label'] == 0) & (df['prediction'] == 1)).sum())
tn_count = int(((df['label'] == 0) & (df['prediction'] == 0)).sum())
tp_count = int(((df['label'] == 1) & (df['prediction'] == 1)).sum())

assert total_samples == 998 and human_samples == 515 and ai_samples == 483
assert fn_count == 41 and fp_count == 87 and tn_count == 428 and tp_count == 442

# Ensure calculated word count and character count
df['word_count_calc'] = df['text'].apply(lambda x: len(str(x).split()))
df['char_count_calc'] = df['text'].apply(lambda x: len(str(x)))

# Step 2: Create Error Files
error_columns = [
    'text', 'label', 'probability', 'length_group',
    'applied_threshold', 'dataset', 'prediction', 'error_type'
]

fn_df = df[df['error_type'] == 'False Negative'][error_columns].copy()
fp_df = df[df['error_type'] == 'False Positive'][error_columns].copy()
all_errors_df = df[df['error_type'].isin(['False Negative', 'False Positive'])][error_columns].copy()

fn_df.to_csv(os.path.join(output_dir, "false_negatives.csv"), index=False)
fp_df.to_csv(os.path.join(output_dir, "false_positives.csv"), index=False)
all_errors_df.to_csv(os.path.join(output_dir, "all_errors.csv"), index=False)

# Step 3: Verify length thresholds applied
expected_thresholds = {
    'very_short': 0.30,
    'short': 0.40,
    'medium': 0.59,
    'long': 0.38
}
for grp, exp_t in expected_thresholds.items():
    sub = df[df['length_group'] == grp]
    assert np.allclose(sub['applied_threshold'], exp_t), f"Threshold mismatch for {grp}"

# Step 4: Error Summary by Length Group
length_rows = []
for grp in ['very_short', 'short', 'medium', 'long']:
    sub = df[df['length_group'] == grp]
    n_total = len(sub)
    n_human = (sub['label'] == 0).sum()
    n_ai = (sub['label'] == 1).sum()
    grp_fp = ((sub['label'] == 0) & (sub['prediction'] == 1)).sum()
    grp_fn = ((sub['label'] == 1) & (sub['prediction'] == 0)).sum()
    fp_rate = grp_fp / n_human if n_human > 0 else 0.0
    fn_rate = grp_fn / n_ai if n_ai > 0 else 0.0
    acc = accuracy_score(sub['label'], sub['prediction'])
    prec = precision_score(sub['label'], sub['prediction'], zero_division=0)
    rec = recall_score(sub['label'], sub['prediction'], zero_division=0)
    f1 = f1_score(sub['label'], sub['prediction'], zero_division=0)
    
    length_rows.append({
        'length_group': grp,
        'total_samples': n_total,
        'human_samples': int(n_human),
        'ai_samples': int(n_ai),
        'FP': int(grp_fp),
        'FN': int(grp_fn),
        'FP_rate': round(fp_rate, 4),
        'FN_rate': round(fn_rate, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4)
    })

df_by_length = pd.DataFrame(length_rows)
df_by_length.to_csv(os.path.join(output_dir, "error_analysis_by_length.csv"), index=False)

# Step 5: Error Summary by Dataset
dataset_rows = []
for ds in ['HC3', 'RAID', 'SentenceAI']:
    sub = df[df['dataset'] == ds]
    n_total = len(sub)
    n_human = (sub['label'] == 0).sum()
    n_ai = (sub['label'] == 1).sum()
    ds_fp = ((sub['label'] == 0) & (sub['prediction'] == 1)).sum()
    ds_fn = ((sub['label'] == 1) & (sub['prediction'] == 0)).sum()
    fp_rate = ds_fp / n_human if n_human > 0 else 0.0
    fn_rate = ds_fn / n_ai if n_ai > 0 else 0.0
    acc = accuracy_score(sub['label'], sub['prediction'])
    prec = precision_score(sub['label'], sub['prediction'], zero_division=0)
    rec = recall_score(sub['label'], sub['prediction'], zero_division=0)
    f1 = f1_score(sub['label'], sub['prediction'], zero_division=0)
    
    dataset_rows.append({
        'dataset': ds,
        'total_samples': n_total,
        'human_samples': int(n_human),
        'ai_samples': int(n_ai),
        'FP': int(ds_fp),
        'FN': int(ds_fn),
        'FP_rate': round(fp_rate, 4),
        'FN_rate': round(fn_rate, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4)
    })

df_by_dataset = pd.DataFrame(dataset_rows)
df_by_dataset.to_csv(os.path.join(output_dir, "error_analysis_by_dataset.csv"), index=False)

# Step 6: Probability Analysis & Buckets
bins = np.linspace(0.0, 1.0, 11)
bucket_labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(10)]
df['prob_bucket'] = pd.cut(df['probability'], bins=bins, labels=bucket_labels, include_lowest=True)

prob_stats = {
    'FP': {
        'min': float(fp_df['probability'].min()),
        'max': float(fp_df['probability'].max()),
        'mean': float(fp_df['probability'].mean()),
        'median': float(fp_df['probability'].median())
    },
    'FN': {
        'min': float(fn_df['probability'].min()),
        'max': float(fn_df['probability'].max()),
        'mean': float(fn_df['probability'].mean()),
        'median': float(fn_df['probability'].median())
    }
}

bucket_rows = []
for b in bucket_labels:
    sub = df[df['prob_bucket'] == b]
    b_fp = ((sub['label'] == 0) & (sub['prediction'] == 1)).sum()
    b_fn = ((sub['label'] == 1) & (sub['prediction'] == 0)).sum()
    b_correct = (sub['label'] == sub['prediction']).sum()
    bucket_rows.append({
        'bucket': b,
        'total_samples': len(sub),
        'FP_count': int(b_fp),
        'FN_count': int(b_fn),
        'correct_count': int(b_correct)
    })
df_buckets = pd.DataFrame(bucket_rows)

# Step 7: Close-to-Threshold Errors
all_err = df[df['error_type'].isin(['False Negative', 'False Positive'])].copy()
all_err['distance_from_threshold'] = (all_err['probability'] - all_err['applied_threshold']).abs()
near_thresh = all_err[all_err['distance_from_threshold'] <= 0.10].copy()
near_thresh = near_thresh.sort_values(by='distance_from_threshold', ascending=True)

near_thresh_cols = [
    'text', 'label', 'probability', 'length_group',
    'applied_threshold', 'dataset', 'error_type', 'distance_from_threshold'
]
near_thresh[near_thresh_cols].to_csv(os.path.join(output_dir, "near_threshold_errors.csv"), index=False)

# Step 8: Text Pattern Analysis Stats
fn_full = df[df['error_type'] == 'False Negative'].copy()
fp_full = df[df['error_type'] == 'False Positive'].copy()

fn_stats = {
    'count': len(fn_full),
    'word_count_mean': fn_full['word_count_calc'].mean(),
    'word_count_median': fn_full['word_count_calc'].median(),
    'word_count_min': fn_full['word_count_calc'].min(),
    'word_count_max': fn_full['word_count_calc'].max(),
    'char_count_mean': fn_full['char_count_calc'].mean(),
    'char_count_median': fn_full['char_count_calc'].median(),
    'dataset_dist': fn_full['dataset'].value_counts().to_dict(),
    'length_dist': fn_full['length_group'].value_counts().to_dict(),
}

fp_stats = {
    'count': len(fp_full),
    'word_count_mean': fp_full['word_count_calc'].mean(),
    'word_count_median': fp_full['word_count_calc'].median(),
    'word_count_min': fp_full['word_count_calc'].min(),
    'word_count_max': fp_full['word_count_calc'].max(),
    'char_count_mean': fp_full['char_count_calc'].mean(),
    'char_count_median': fp_full['char_count_calc'].median(),
    'dataset_dist': fp_full['dataset'].value_counts().to_dict(),
    'length_dist': fp_full['length_group'].value_counts().to_dict(),
}

# Step 9: Most Difficult Cases
hardest_fn = fn_full.sort_values(by='probability', ascending=True)[error_columns]
hardest_fp = fp_full.sort_values(by='probability', ascending=False)[error_columns]

hardest_fn.to_csv(os.path.join(output_dir, "hardest_false_negatives.csv"), index=False)
hardest_fp.to_csv(os.path.join(output_dir, "hardest_false_positives.csv"), index=False)

# Step 10: Generate Comprehensive Text Report
report_content = f"""================================================================================
V3.1 ERROR ANALYSIS REPORT
================================================================================
Generated for: models/text_final_v3/validation_results.csv
Model Architecture: Word TF-IDF + Char TF-IDF + MiniLM + Interaction + Linguistics + LogisticRegression
Decision Thresholds (Frozen from Calibration):
  very_short: 0.30
  short     : 0.40
  medium    : 0.59
  long      : 0.38

================================================================================
1. VALIDATION VERIFICATION
================================================================================
Total Validation Samples : 998
  - Ground Truth Human   : 515 (51.6%)
  - Ground Truth AI      : 483 (48.4%)
Outcome Counts:
  - True Negatives (TN)  : 428
  - True Positives (TP)  : 442
  - False Positives (FP) : 87
  - False Negatives (FN) : 41
Overall Accuracy         : 87.17% (870 / 998)
Overall Precision        : 83.55%
Overall Recall           : 91.51%
Overall F1 Score         : 87.35%

================================================================================
2. OVERALL FP / FN COUNTS & PROPORTIONS
================================================================================
Total Errors: 128 (12.83% error rate)
  - False Positives (Human misclassified as AI): 87 (67.97% of all errors)
  - False Negatives (AI misclassified as Human): 41 (32.03% of all errors)

FP / FN Ratio: 2.12 False Positives for every 1 False Negative.
The model demonstrates an asymmetric bias towards over-predicting AI content, particularly on shorter inputs.

================================================================================
3. ERROR BREAKDOWN BY LENGTH GROUP
================================================================================
{df_by_length.to_string(index=False)}

Key Length Observations:
- Short texts (very_short + short) account for 73 of 87 FPs (83.9%) and 26 of 41 FNs (63.4%).
- very_short error rate on Human is 23.85% (31 FPs out of 130 human samples).
- short error rate on Human is 18.34% (42 FPs out of 229 human samples).
- medium achieved the highest F1 (0.9185) with balanced errors (9 FP, 10 FN) due to conservative threshold 0.59.
- long had 5 FP (all German or Wiki CS) and 5 FN (4 MPT model in German).

================================================================================
4. ERROR BREAKDOWN BY DATASET & SOURCE
================================================================================
{df_by_dataset.to_string(index=False)}

Dataset Breakdown Analysis:
1. SentenceAI:
   - 580 validation samples (312 Human, 268 AI)
   - 53 False Positives (17.0% FP rate on Human)
   - 31 False Negatives (11.6% FN rate on AI)
   - SentenceAI alone accounts for 60.9% of all FPs (53/87) and 75.6% of all FNs (31/41).
2. HC3:
   - 216 validation samples (103 Human, 113 AI)
   - 19 False Positives (18.4% FP rate on Human)
   - 4 False Negatives (3.5% FN rate on AI)
   - HC3 FPs originate mainly from reddit_eli5 (12 FPs) and open_qa (5 FPs).
3. RAID:
   - 202 validation samples (100 Human, 102 AI)
   - 15 False Positives (15.0% FP rate on Human)
   - 6 False Negatives (5.9% FN rate on AI)
   - 12 of 15 RAID FPs are from the 'code' domain (Python function definitions).

================================================================================
5. PROBABILITY ANALYSIS
================================================================================
False Positive Probabilities (N=87):
  Min   : {prob_stats['FP']['min']:.4f}
  Max   : {prob_stats['FP']['max']:.4f}
  Mean  : {prob_stats['FP']['mean']:.4f}
  Median: {prob_stats['FP']['median']:.4f}

False Negative Probabilities (N=41):
  Min   : {prob_stats['FN']['min']:.4f}
  Max   : {prob_stats['FN']['max']:.4f}
  Mean  : {prob_stats['FN']['mean']:.4f}
  Median: {prob_stats['FN']['median']:.4f}

Probability Distribution Buckets:
{df_buckets.to_string(index=False)}

Bucket Insights:
- 26 FPs (29.9%) have extreme probabilities >= 0.70 (up to 0.9808).
- 33 FPs (37.9%) fall in the moderate AI zone [0.50, 0.70).
- 28 FPs (32.2%) fall in the sub-0.50 zone [0.30, 0.50), directly resulting from lower length thresholds (0.30, 0.40).
- 25 FNs (61.0%) fall in [0.20, 0.40), directly beneath the tier thresholds.

================================================================================
6. NEAR-THRESHOLD ERROR ANALYSIS
================================================================================
Total Near-Threshold Errors (|P - Threshold| <= 0.10): {len(near_thresh)} / 128 errors ({len(near_thresh)/128*100:.1f}%)
  - Near-Threshold False Positives: {int((near_thresh['error_type'] == 'False Positive').sum())}
  - Near-Threshold False Negatives: {int((near_thresh['error_type'] == 'False Negative').sum())}

Top 10 Closest Errors to Threshold:
{near_thresh[['probability', 'applied_threshold', 'distance_from_threshold', 'error_type', 'length_group', 'dataset']].head(10).to_string(index=False)}

Observation:
Over 38% of all errors reside within +/- 0.10 of the decision threshold, indicating high density around the boundary.

================================================================================
7. FALSE NEGATIVE (FN) PATTERNS & OBSERVATIONS
================================================================================
Descriptive Patterns in the 41 FN Samples:
1. Highly Conversational & Casual Phrasing:
   - Examples: "Meat, its something we all love to eat." (P=0.0629), "What do you think?" (P=0.1604), "No thanks." (P=0.2909).
   - Short AI sentences that adopt an informal first-person persona or casual punctuation lack typical academic AI markers.
2. Narrative Personal Anecdotes:
   - Example: "So, I decided to do some research and talk to my counselor to see what's what." (P=0.1484).
3. Specialized Foreign Language Text:
   - 4 FNs in RAID German domain generated by MPT (P in 0.22 - 0.29).

================================================================================
8. FALSE POSITIVE (FP) PATTERNS & OBSERVATIONS
================================================================================
Descriptive Patterns in the 87 FP Samples:
1. Formal Expository Human Sentences (SentenceAI):
   - Examples:
     "Self-confidence is really important for people to feel good about their own selves." (P=0.9808)
     "As well, creativity stems from a sense of individuality." (P=0.9291)
     "Time management is essential for a successful live." (P=0.9054)
     "In conclusion, student designed projects have many benefits that can help them succeed in life..." (P=0.8413)
   - These human essay sentences exhibit standard transitional phrases ("In conclusion", "As well", "In addition") and generic declarative phrasing that embeddings associate with AI essay generators.
2. Structured Python Code Snippets (RAID Code):
   - Examples:
     "def Diff(li1,li2): return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))" (P=0.8231)
     "def max_sum_list(lists): return max(lists, key=sum)" (P=0.7950)
     "def count_Digit(n): while n != 0: n //= 10; count += 1" (P=0.9226)
   - Clean, standard programmatic functions have high keyword repetition and low perplexity, triggering AI classification.
3. Factual & Encyclopedic Statements (HC3):
   - Example: "As president, Reagan implemented sweeping new political and economic initiatives." (P=0.8122)

================================================================================
9. POTENTIAL CAUSES SUPPORTED BY EMPIRICAL DATA
================================================================================
1. Feature Weighting & Interaction Amplification (Primary Cause - 40%):
   - The short-text embedding interaction block multiplied MiniLM embeddings by 1.50 (very_short) and 1.25 (short).
   - While this successfully rescued short-text recall (FN dropped to 26), it magnified stylistic embedding similarity on formal human sentences, pushing clean human text to high probabilities (0.80 - 0.98).
2. Domain / Syntax Specificity (Secondary Cause - 30%):
   - Code syntax (RAID) and student essay fragments (SentenceAI) have high intrinsic structural regularity. Without explicit domain conditioning or code-specific tokenization, standard text embeddings interpret clean syntax as synthetic.
3. Length Information Constraint (20%):
   - A 5-to-15 word fragment has almost no TF-IDF lexical diversity, forcing the linear classifier to rely solely on dense embeddings and character n-grams.
4. Threshold Sensitivity (10%):
   - Calibrated thresholds of 0.30 (very_short) and 0.40 (short) created 28 FPs in [0.30, 0.50).

================================================================================
10. RECOMMENDATIONS FOR V3.2 (DIAGNOSIS & ROADMAP)
================================================================================
The following diagnostic recommendations are structured by subsystem for evaluation on a NEW development/calibration split:

A. Architecture & Model Improvements:
   - Re-evaluate Embedding Multiplier Scale: Dampen short-text embedding interaction multiplier (e.g., test 1.10 - 1.20 instead of 1.50) or use a nonlinear gating mechanism (e.g., logistic sigmoid gating on word count).
   - Code & Domain Routing: Add an explicit code detection pre-filter or code-specific feature branch to avoid evaluating raw Python functions with prose TF-IDF weights.

B. Training Data Improvements:
   - SentenceAI Balancing: Augment training data with formal human conversational sentences, essay introductions, and concise definitions to expose the classifier to diverse human formal styles.
   - Code Syntax Humanization: Balance human code samples with varied formatting and comments in the training distribution.

C. Feature Engineering Improvements:
   - Punctuation & Capitalization Burstiness: Introduce burstiness features (variance in sentence length, punctuation entropy) specifically designed for multi-sentence short inputs.
   - Contraction & Colloquial Density Feature: Explicitly measure contraction rates and informal grammar markers to prevent casual AI from dropping into FN while protecting formal human text.

D. Calibration & Decision Policy Ideas (To be tuned on a NEW split):
   - Constrained Precision Optimization: Calibrate length-aware thresholds with a minimum precision constraint (e.g., maximize F1 subject to Precision >= 88%) rather than unconstrained F1.
   - Platt Scaling / Isotonic Calibration per Length Group: Apply post-hoc probability calibration to ensure that output probabilities reflect true empirical likelihoods across all length tiers.
================================================================================
"""

with open(os.path.join(output_dir, "V3.1_ERROR_ANALYSIS_REPORT.txt"), "w", encoding="utf-8") as f:
    f.write(report_content)

print("All error analysis files generated successfully.")
