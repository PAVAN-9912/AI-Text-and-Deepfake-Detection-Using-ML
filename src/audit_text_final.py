import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# Set working directory to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print("=" * 80)
print("PHASE A — AUDIT OF TEXT DETECTION IMPLEMENTATION & DATA HYGIENE")
print("=" * 80)

# 1. Dataset Paths
HC3_TRAIN = os.path.join(BASE_DIR, "datasets", "text", "HC3", "train.csv")
HC3_VALIDATION = os.path.join(BASE_DIR, "datasets", "text", "HC3", "validation.csv")
RAID_TRAIN = os.path.join(BASE_DIR, "datasets", "text", "RAID", "raid_domain_training_sample.csv")
SENTENCE_TRAIN = os.path.join(BASE_DIR, "datasets", "text", "SentenceAI", "sentence_ai_training.csv")

# External datasets
RAID_EXTERNAL = os.path.join(BASE_DIR, "datasets", "text", "RAID", "raid_external.csv")
SENTENCE_TEST = os.path.join(BASE_DIR, "datasets", "text", "SentenceAI", "sentence_ai_test.csv")
REAL_WORLD_TEST = os.path.join(BASE_DIR, "datasets", "text", "realworld_test.csv")
AIGCODESET_TEST = os.path.join(BASE_DIR, "datasets", "text", "CodeAI", "aigcodeset_test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models", "text_final")

# Check file existence
files_to_check = {
    "HC3 Train": HC3_TRAIN,
    "HC3 Validation": HC3_VALIDATION,
    "RAID Train Sample": RAID_TRAIN,
    "SentenceAI Train": SENTENCE_TRAIN,
    "RAID External Test": RAID_EXTERNAL,
    "SentenceAI Test": SENTENCE_TEST,
    "Real-World Test": REAL_WORLD_TEST,
    "AIGCodeSet Test": AIGCODESET_TEST,
}

print("\n1. DATASET FILE EXISTENCE & RECORD COUNTS:")
print("-" * 50)
for name, path in files_to_check.items():
    exists = os.path.exists(path)
    if exists:
        try:
            df = pd.read_csv(path)
            print(f"  [OK] {name:20s}: {len(df):6d} rows | Path: {os.path.relpath(path, BASE_DIR)}")
        except Exception as e:
            print(f"  [WARN] {name:20s}: Exists but read error ({e})")
    else:
        print(f"  [MISSING] {name:20s}: Path: {os.path.relpath(path, BASE_DIR)}")

# 2. Check Model Artifacts
print("\n2. MODEL ARTIFACT INTEGRITY CHECK (models/text_final/):")
print("-" * 50)
artifacts = [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "multi_char_tfidf_vectorizer.pkl",
    "linguistic_scaler.pkl",
    "text_classifier.pkl",
    "probability_calibrator.pkl",
    "text_config.pkl",
    "text_metrics.pkl"
]

for art in artifacts:
    p = os.path.join(MODEL_DIR, art)
    if os.path.exists(p):
        size_kb = os.path.getsize(p) / 1024
        print(f"  [OK] {art:32s}: {size_kb:8.1f} KB")
    else:
        print(f"  [FAIL] {art:32s}: MISSING")

# 3. Load config and metrics
with open(os.path.join(MODEL_DIR, "text_config.pkl"), "rb") as f:
    config = pickle.load(f)
with open(os.path.join(MODEL_DIR, "text_metrics.pkl"), "rb") as f:
    metrics = pickle.load(f)

print("\n3. STORED CONFIGURATION & METRICS AUDIT:")
print("-" * 50)
print(f"  Model Version       : {config.get('model_version')}")
print(f"  Word Max Features   : {config.get('word_max_features')}")
print(f"  Char Max Features   : {config.get('char_max_features')}")
print(f"  Multi-Char Max Feat : {config.get('multi_char_max_features')}")
print(f"  Scalar Features     : {config.get('total_scalar_features')} features")
print(f"  Short Decay Alpha   : {config.get('short_interaction_alpha')}")
print(f"  Classifier C        : {config.get('classifier_C')}")
print(f"  Decision Thresholds : {config.get('length_thresholds')}")
print(f"  Train Samples       : {config.get('training_samples')}")
print(f"  Calibration Samples : {config.get('calibration_samples')}")
print(f"  Validation Samples  : {config.get('validation_samples')}")

# 4. Split Reconstruction & Overlap Audit
print("\n4. RECONSTRUCTING DATASET SPLITS & CHECKING ZERO-LEAKAGE HYGIENE:")
print("-" * 50)

def clean_dataframe(df):
    text_col = "text" if "text" in df.columns else ("content" if "content" in df.columns else ("sentence" if "sentence" in df.columns else None))
    if text_col is None:
        return pd.DataFrame()
    df = df.copy()
    if text_col != "text":
        df["text"] = df[text_col]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    return df

def get_length_group(text):
    wc = len(str(text).split())
    if wc <= 10: return "very_short"
    elif wc <= 30: return "short"
    elif wc <= 60: return "medium"
    else: return "long"

def sample_length_label(df, label, length_group, target, seed=42):
    subset = df[(df["label"] == label) & (df["length_group"] == length_group)].copy()
    if len(subset) == 0: return pd.DataFrame()
    if len(subset) >= target: return subset.sample(n=target, random_state=seed)
    return subset

hc3_tr = clean_dataframe(pd.read_csv(HC3_TRAIN))
hc3_val = clean_dataframe(pd.read_csv(HC3_VALIDATION))
hc3_tr["length_group"] = hc3_tr["text"].apply(get_length_group)
hc3_val["length_group"] = hc3_val["text"].apply(get_length_group)

raid_tr = clean_dataframe(pd.read_csv(RAID_TRAIN))
raid_tr["length_group"] = raid_tr["text"].apply(get_length_group)

sent_tr = clean_dataframe(pd.read_csv(SENTENCE_TRAIN))
sent_tr["label"] = sent_tr["label_name"].map({"Human": 0, "AI": 1})
sent_tr = sent_tr[sent_tr["label"].isin([0, 1])].copy()
sent_tr["length_group"] = sent_tr["text"].apply(get_length_group)

TARGETS = {"very_short": 1500, "short": 2500, "medium": 1500, "long": 500}
selected_parts = []
for length_group in ["very_short", "short", "medium", "long"]:
    target = TARGETS[length_group]
    sentence_target = 0
    if length_group == "very_short": sentence_target = min(1000, target)
    elif length_group == "short": sentence_target = min(1500, target)
    elif length_group == "medium": sentence_target = min(500, target)

    if sentence_target > 0:
        for label in [0, 1]:
            s = sample_length_label(sent_tr, label, length_group, sentence_target)
            if len(s) > 0:
                s["dataset"] = "SentenceAI"
                selected_parts.append(s)

    remaining = target - sentence_target
    hc3_target = remaining // 2
    raid_target = remaining - hc3_target

    for label in [0, 1]:
        s = sample_length_label(hc3_tr, label, length_group, hc3_target)
        if len(s) > 0:
            s["dataset"] = "HC3"
            selected_parts.append(s)

    for label in [0, 1]:
        subset = raid_tr[(raid_tr["label"] == label) & (raid_tr["length_group"] == length_group)].copy()
        if label == 1 and len(subset) > 0 and "raid_model" in subset.columns:
            generators = ["gpt2", "llama-chat", "mpt", "mpt-chat"]
            per_gen = max(1, raid_target // len(generators))
            gen_parts = []
            for g in generators:
                gdf = subset[subset["raid_model"] == g]
                if len(gdf) > 0:
                    gen_parts.append(gdf.sample(n=min(per_gen, len(gdf)), random_state=42))
            if gen_parts:
                s = pd.concat(gen_parts, ignore_index=True)
                if len(s) < raid_target:
                    chosen = set(s["text"])
                    rem = subset[~subset["text"].isin(chosen)]
                    if len(rem) > 0:
                        extra = rem.sample(n=min(raid_target - len(s), len(rem)), random_state=42)
                        s = pd.concat([s, extra], ignore_index=True)
            else:
                s = subset.sample(n=min(raid_target, len(subset)), random_state=42)
        else:
            s = subset.sample(n=min(raid_target, len(subset)), random_state=42) if len(subset) > 0 else pd.DataFrame()
        if len(s) > 0:
            s["dataset"] = "RAID"
            selected_parts.append(s)

full_train_df = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
full_train_df = full_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 10% held-out validation
val_parts = []
for dname in ["HC3", "RAID", "SentenceAI"]:
    ddf = full_train_df[full_train_df["dataset"] == dname].copy()
    for lval in sorted(ddf["label"].unique()):
        for lgval in ["very_short", "short", "medium", "long"]:
            gdf = ddf[(ddf["label"] == lval) & (ddf["length_group"] == lgval)].copy()
            if len(gdf) == 0: continue
            take = max(1, int(len(gdf) * 0.10))
            take = min(take, len(gdf))
            val_parts.append(gdf.sample(n=take, random_state=42).copy())

validation_df = pd.concat(val_parts, ignore_index=True).reset_index(drop=True)
val_texts = set(validation_df["text"])
remaining_train_df = full_train_df[~full_train_df["text"].isin(val_texts)].reset_index(drop=True)

from sklearn.model_selection import train_test_split
strat_key = remaining_train_df["label"].astype(str) + "_" + remaining_train_df["length_group"].astype(str)
core_idx, cal_idx = train_test_split(
    np.arange(len(remaining_train_df)),
    test_size=0.10,
    random_state=42,
    stratify=strat_key
)
model_train_df = remaining_train_df.iloc[core_idx].reset_index(drop=True)
calibration_df = remaining_train_df.iloc[cal_idx].reset_index(drop=True)

print(f"  Model Training Set : {len(model_train_df)} samples (Expected: 8160)")
print(f"  Calibration Set    : {len(calibration_df)} samples (Expected: 907)")
print(f"  Validation Set     : {len(validation_df)} samples (Expected: 998)")

# Overlap checks
train_set = set(model_train_df["text"])
cal_set = set(calibration_df["text"])
val_set = set(validation_df["text"])

train_cal_overlap = len(train_set.intersection(cal_set))
train_val_overlap = len(train_set.intersection(val_set))
cal_val_overlap = len(cal_set.intersection(val_set))

print(f"\n  Overlap Audit:")
print(f"  - Train & Calibration Overlap : {train_cal_overlap} (Strict Zero Required)")
print(f"  - Train & Validation Overlap  : {train_val_overlap} (Strict Zero Required)")
print(f"  - Cal & Validation Overlap    : {cal_val_overlap} (Strict Zero Required)")

# External dataset overlap check
if os.path.exists(RAID_EXTERNAL):
    raid_ext = clean_dataframe(pd.read_csv(RAID_EXTERNAL))
    raid_ext_set = set(raid_ext["text"])
    ext_overlap = len(train_set.intersection(raid_ext_set)) + len(cal_set.intersection(raid_ext_set)) + len(val_set.intersection(raid_ext_set))
    print(f"  - RAID External & Development Overlap : {ext_overlap}")

if os.path.exists(SENTENCE_TEST):
    sent_ext = clean_dataframe(pd.read_csv(SENTENCE_TEST))
    sent_ext_set = set(sent_ext["text"])
    ext_sent_overlap = len(train_set.intersection(sent_ext_set)) + len(cal_set.intersection(sent_ext_set)) + len(val_set.intersection(sent_ext_set))
    print(f"  - SentenceAI Test & Development Overlap: {ext_sent_overlap}")

assert train_cal_overlap == 0 and train_val_overlap == 0 and cal_val_overlap == 0, "Data leakage detected!"
print("\n>>> AUDIT PASSED: ZERO DATA LEAKAGE CONFIRMED ACROSS ALL SPLITS. <<<")
