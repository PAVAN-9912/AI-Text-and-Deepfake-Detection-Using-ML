import os
import pickle
import random
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("V3.2 EXPERIMENT: SHORT-TEXT REPRESENTATION & REGULARIZATION")
print("=" * 70)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================================
# DIRECTORY & PATH SETTINGS
# =========================================================

HC3_TRAIN = r"datasets\text\HC3\train.csv"
HC3_VALIDATION = r"datasets\text\HC3\validation.csv"
RAID_TRAIN = r"datasets\text\RAID\raid_domain_training_sample.csv"
SENTENCE_TRAIN = r"datasets\text\SentenceAI\sentence_ai_training.csv"

V3_MODEL_DIR = r"models\text_final_v3"
V32_MODEL_DIR = r"models\text_final_v3_2"
V32_ERROR_DIR = os.path.join(V32_MODEL_DIR, "error_analysis")

os.makedirs(V32_MODEL_DIR, exist_ok=True)
os.makedirs(V32_ERROR_DIR, exist_ok=True)

WORD_MAX_FEATURES = 150000
CHAR_MAX_FEATURES = 150000

TARGETS = {
    "very_short": 1500,
    "short": 2500,
    "medium": 1500,
    "long": 500
}

FINAL_VALIDATION_FRACTION = 0.10
CALIBRATION_FRACTION = 0.10
TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_text_column(df):
    if "text" in df.columns:
        return "text"
    if "content" in df.columns:
        return "content"
    if "sentence" in df.columns:
        return "sentence"
    return None

def clean_dataframe(df):
    text_col = get_text_column(df)
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
    if wc <= 10:
        return "very_short"
    elif wc <= 30:
        return "short"
    elif wc <= 60:
        return "medium"
    else:
        return "long"

def add_length_group(df):
    df = df.copy()
    df["length_group"] = df["text"].apply(get_length_group)
    return df

def sample_length_label(df, label, length_group, target, dataset_name):
    subset = df[(df["label"] == label) & (df["length_group"] == length_group)].copy()
    available = len(subset)
    if available == 0:
        return pd.DataFrame()
    if available >= target:
        return subset.sample(n=target, random_state=RANDOM_SEED)
    return subset

def build_linguistic_features(texts):
    features = []
    for text in texts:
        text = str(text)
        words = text.split()
        word_count = len(words)

        sentence_count = sum(1 for char in text if char in ".!?")
        if sentence_count == 0:
            sentence_count = 1

        avg_sentence_length = word_count / max(sentence_count, 1)

        if word_count > 0:
            unique_words = len(set(word.lower() for word in words))
            vocabulary_diversity = unique_words / word_count
        else:
            vocabulary_diversity = 0.0

        log_word_count = np.log1p(word_count)

        if word_count <= 10:
            length_bucket = 0.0
        elif word_count <= 30:
            length_bucket = 1.0
        elif word_count <= 60:
            length_bucket = 2.0
        else:
            length_bucket = 3.0

        features.append([
            log_word_count,
            sentence_count,
            avg_sentence_length,
            vocabulary_diversity,
            length_bucket
        ])
    return np.asarray(features, dtype=np.float32)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# =========================================================
# DATASET LOADING (EXACT REPLICATION OF V3/V3.1)
# =========================================================

print("\n1. LOADING DATASETS")
print("-------------------")

hc3_train = clean_dataframe(pd.read_csv(HC3_TRAIN))
hc3_validation = clean_dataframe(pd.read_csv(HC3_VALIDATION))
hc3_train = add_length_group(hc3_train)
hc3_validation = add_length_group(hc3_validation)

raid = clean_dataframe(pd.read_csv(RAID_TRAIN))
raid = add_length_group(raid)

sentence_train = clean_dataframe(pd.read_csv(SENTENCE_TRAIN))
sentence_train["label"] = sentence_train["label_name"].map({"Human": 0, "AI": 1})
sentence_train = sentence_train[sentence_train["label"].isin([0, 1])].copy()
sentence_train = add_length_group(sentence_train)

# Build length-aware training data
selected_parts = []
for length_group in ["very_short", "short", "medium", "long"]:
    target = TARGETS[length_group]

    sentence_target = 0
    if length_group == "very_short":
        sentence_target = min(1000, target)
    elif length_group == "short":
        sentence_target = min(1500, target)
    elif length_group == "medium":
        sentence_target = min(500, target)

    if sentence_target > 0:
        for label in [0, 1]:
            sampled = sample_length_label(sentence_train, label, length_group, sentence_target, "SentenceAI")
            if len(sampled) > 0:
                sampled = sampled.copy()
                sampled["dataset"] = "SentenceAI"
                selected_parts.append(sampled)

    remaining_target = target - sentence_target
    hc3_target = remaining_target // 2
    raid_target = remaining_target - hc3_target

    # HC3
    for label in [0, 1]:
        sampled = sample_length_label(hc3_train, label, length_group, hc3_target, "HC3")
        if len(sampled) > 0:
            sampled = sampled.copy()
            sampled["dataset"] = "HC3"
            selected_parts.append(sampled)

    # RAID
    for label in [0, 1]:
        subset = raid[(raid["label"] == label) & (raid["length_group"] == length_group)].copy()
        if label == 1 and len(subset) > 0 and "raid_model" in subset.columns:
            generators = ["gpt2", "llama-chat", "mpt", "mpt-chat"]
            per_generator = max(1, raid_target // len(generators))
            generator_parts = []
            for generator in generators:
                generator_df = subset[subset["raid_model"] == generator]
                if len(generator_df) == 0:
                    continue
                take = min(per_generator, len(generator_df))
                generator_parts.append(generator_df.sample(n=take, random_state=RANDOM_SEED))
            if generator_parts:
                sampled = pd.concat(generator_parts, ignore_index=True)
                if len(sampled) < raid_target:
                    selected_texts = set(sampled["text"])
                    remaining_pool = subset[~subset["text"].isin(selected_texts)]
                    extra_needed = raid_target - len(sampled)
                    if extra_needed > 0 and len(remaining_pool) > 0:
                        extra = remaining_pool.sample(n=min(extra_needed, len(remaining_pool)), random_state=RANDOM_SEED)
                        sampled = pd.concat([sampled, extra], ignore_index=True)
            else:
                sampled = subset.sample(n=min(raid_target, len(subset)), random_state=RANDOM_SEED)
        else:
            if len(subset) > 0:
                sampled = subset.sample(n=min(raid_target, len(subset)), random_state=RANDOM_SEED)
            else:
                sampled = pd.DataFrame()
        if len(sampled) > 0:
            sampled = sampled.copy()
            sampled["dataset"] = "RAID"
            selected_parts.append(sampled)

train_df = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
train_df = train_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# Held-out validation split
validation_parts = []
for dataset_name in ["HC3", "RAID", "SentenceAI"]:
    dataset_df = train_df[train_df["dataset"] == dataset_name].copy()
    if len(dataset_df) == 0:
        continue
    for label_value in sorted(dataset_df["label"].unique()):
        for length_value in ["very_short", "short", "medium", "long"]:
            group_df = dataset_df[(dataset_df["label"] == label_value) & (dataset_df["length_group"] == length_value)].copy()
            if len(group_df) == 0:
                continue
            take = max(1, int(len(group_df) * FINAL_VALIDATION_FRACTION))
            take = min(take, len(group_df))
            sampled = group_df.sample(n=take, random_state=RANDOM_SEED).copy()
            validation_parts.append(sampled)

validation_df = pd.concat(validation_parts, ignore_index=True).reset_index(drop=True)
validation_texts = set(validation_df["text"])
train_df = train_df[~train_df["text"].isin(validation_texts)].reset_index(drop=True)

# Calibration split from remaining train
stratification_key = train_df["label"].astype(str) + "_" + train_df["length_group"].astype(str)
try:
    core_idx, calibration_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=stratification_key
    )
except ValueError:
    core_idx, calibration_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_SEED
    )

model_train_df = train_df.iloc[core_idx].reset_index(drop=True)
calibration_df = train_df.iloc[calibration_idx].reset_index(drop=True)

print(f"Dataset Splits Verified: Train={len(model_train_df)}, Calibration={len(calibration_df)}, Validation={len(validation_df)}")
assert len(model_train_df) == 8160 and len(calibration_df) == 907 and len(validation_df) == 998

# =========================================================
# FEATURE EXTRACTION (COMPUTED ONCE & CACHED)
# =========================================================

print("\n2. EXTRACTING BASE FEATURES (ONCE)")
print("----------------------------------")

# Word TF-IDF
print("Fitting Word TF-IDF...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=WORD_MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    max_df=0.98
)
X_train_word = word_vectorizer.fit_transform(model_train_df["text"])
X_calibration_word = word_vectorizer.transform(calibration_df["text"])
X_validation_word = word_vectorizer.transform(validation_df["text"])

# Char TF-IDF
print("Fitting Character TF-IDF...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    max_df=0.98
)
X_train_char = char_vectorizer.fit_transform(model_train_df["text"])
X_calibration_char = char_vectorizer.transform(calibration_df["text"])
X_validation_char = char_vectorizer.transform(validation_df["text"])

# MiniLM Embeddings
print("Encoding MiniLM Embeddings...")
transformer_model = SentenceTransformer(TRANSFORMER_NAME)
X_train_embedding = transformer_model.encode(model_train_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_calibration_embedding = transformer_model.encode(calibration_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_validation_embedding = transformer_model.encode(validation_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

# Linguistic Features
print("Building Linguistic Features...")
X_train_ling_raw = build_linguistic_features(model_train_df["text"].tolist())
X_cal_ling_raw = build_linguistic_features(calibration_df["text"].tolist())
X_val_ling_raw = build_linguistic_features(validation_df["text"].tolist())

linguistic_scaler = StandardScaler()
X_train_ling = linguistic_scaler.fit_transform(X_train_ling_raw)
X_calibration_ling = linguistic_scaler.transform(X_cal_ling_raw)
X_validation_ling = linguistic_scaler.transform(X_val_ling_raw)

# Base sparse matrices
X_train_emb_sparse = csr_matrix(X_train_embedding)
X_cal_emb_sparse = csr_matrix(X_calibration_embedding)
X_val_emb_sparse = csr_matrix(X_validation_embedding)

X_train_ling_sparse = csr_matrix(X_train_ling)
X_cal_ling_sparse = csr_matrix(X_calibration_ling)
X_val_ling_sparse = csr_matrix(X_validation_ling)

# Labels & Sample Weights
y_train = model_train_df["label"].values.astype(int)
y_calibration = calibration_df["label"].values.astype(int)
y_validation = validation_df["label"].values.astype(int)

length_weights = {"very_short": 1.25, "short": 1.15, "medium": 1.05, "long": 1.00}
train_sample_weights = model_train_df["length_group"].map(length_weights).astype(float).values

# Word counts for continuous shortness factors
train_word_counts = np.array([len(t.split()) for t in model_train_df["text"]], dtype=np.float32)
cal_word_counts = np.array([len(t.split()) for t in calibration_df["text"]], dtype=np.float32)
val_word_counts = np.array([len(t.split()) for t in validation_df["text"]], dtype=np.float32)

# =========================================================
# DEFINING CANDIDATE INTERACTION STRATEGIES
# =========================================================

def build_interaction_block(strategy_type, param, emb, df, word_counts):
    if strategy_type == "v31_baseline":
        mult = df["length_group"].map(param).astype(np.float32).values.reshape(-1, 1)
        return csr_matrix((emb * mult).astype(np.float32))

    elif strategy_type == "strat1_moderated":
        mult = df["length_group"].map(param).astype(np.float32).values.reshape(-1, 1)
        return csr_matrix((emb * mult).astype(np.float32))

    elif strategy_type == "strat2_exp_decay":
        alpha = float(param)
        decay = (alpha * np.exp(-word_counts / 15.0)).reshape(-1, 1).astype(np.float32)
        return csr_matrix((emb * decay).astype(np.float32))

    elif strategy_type == "strat3_explicit_shortness":
        alpha = float(param)
        decay = (alpha * np.exp(-word_counts / 15.0)).reshape(-1, 1).astype(np.float32)
        decay_emb = emb * decay
        f1 = np.exp(-word_counts / 15.0).reshape(-1, 1).astype(np.float32)
        f2 = (1.0 / (1.0 + word_counts)).reshape(-1, 1).astype(np.float32)
        combined_extra = np.hstack([decay_emb, f1, f2])
        return csr_matrix(combined_extra.astype(np.float32))

    raise ValueError(f"Unknown strategy: {strategy_type}")

# =========================================================
# CALIBRATION THRESHOLD SEARCH (STABLE SELECTION)
# =========================================================

def find_stable_length_thresholds(y_cal, probs_cal, cal_df):
    length_thresholds = {}
    sweep_results = []
    
    for group in ["very_short", "short", "medium", "long"]:
        mask = (cal_df["length_group"] == group).values
        y_g = y_cal[mask]
        p_g = probs_cal[mask]
        
        group_sweeps = []
        for t in np.arange(0.20, 0.81, 0.01):
            t_val = round(float(t), 2)
            preds = (p_g >= t_val).astype(int)
            f1 = f1_score(y_g, preds, zero_division=0)
            acc = accuracy_score(y_g, preds)
            prec = precision_score(y_g, preds, zero_division=0)
            rec = recall_score(y_g, preds, zero_division=0)
            group_sweeps.append({
                "threshold": t_val,
                "f1": f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            })
            sweep_results.append({
                "length_group": group,
                "threshold": t_val,
                "f1": f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            })
            
        df_sw = pd.DataFrame(group_sweeps)
        max_f1 = df_sw["f1"].max()
        
        # Stable plateau selection (near-optimal within 0.005)
        near_optimal = df_sw[df_sw["f1"] >= (max_f1 - 0.005)]
        chosen_thresh = float(near_optimal["threshold"].median())
        chosen_thresh = round(chosen_thresh, 2)
        length_thresholds[group] = chosen_thresh
        
    return length_thresholds, sweep_results

# =========================================================
# EXPERIMENTATION ON TRAINING & CALIBRATION ONLY
# =========================================================

print("\n3. RUNNING CONTROLLED EXPERIMENTS (TRAIN & CALIBRATION ONLY)")
print("----------------------------------------------------------")

candidate_configs = [
    # Baseline V3.1 Reference
    {
        "id": "V3.1_Reference",
        "strategy": "v31_baseline",
        "param": {"very_short": 1.50, "short": 1.25, "medium": 0.75, "long": 0.50},
        "C": 2.0,
        "desc": "Baseline V3.1 Multipliers [1.50, 1.25, 0.75, 0.50], C=2.0"
    },
    # Strategy 1: Moderated Discrete Multipliers
    {
        "id": "Strat1_Mod_C1.0",
        "strategy": "strat1_moderated",
        "param": {"very_short": 1.15, "short": 1.10, "medium": 0.85, "long": 0.70},
        "C": 1.0,
        "desc": "Strat 1 Moderated Multipliers [1.15, 1.10, 0.85, 0.70], C=1.0"
    },
    {
        "id": "Strat1_Mod_C2.0",
        "strategy": "strat1_moderated",
        "param": {"very_short": 1.15, "short": 1.10, "medium": 0.85, "long": 0.70},
        "C": 2.0,
        "desc": "Strat 1 Moderated Multipliers [1.15, 1.10, 0.85, 0.70], C=2.0"
    },
    {
        "id": "Strat1_Mod_C3.0",
        "strategy": "strat1_moderated",
        "param": {"very_short": 1.15, "short": 1.10, "medium": 0.85, "long": 0.70},
        "C": 3.0,
        "desc": "Strat 1 Moderated Multipliers [1.15, 1.10, 0.85, 0.70], C=3.0"
    },
    # Strategy 2: Continuous Exponential Decay
    {
        "id": "Strat2_Exp_a0.25_C2.0",
        "strategy": "strat2_exp_decay",
        "param": 0.25,
        "C": 2.0,
        "desc": "Strat 2 Continuous Decay: alpha=0.25, C=2.0"
    },
    {
        "id": "Strat2_Exp_a0.50_C1.0",
        "strategy": "strat2_exp_decay",
        "param": 0.50,
        "C": 1.0,
        "desc": "Strat 2 Continuous Decay: alpha=0.50, C=1.0"
    },
    {
        "id": "Strat2_Exp_a0.50_C2.0",
        "strategy": "strat2_exp_decay",
        "param": 0.50,
        "C": 2.0,
        "desc": "Strat 2 Continuous Decay: alpha=0.50, C=2.0"
    },
    {
        "id": "Strat2_Exp_a0.50_C3.0",
        "strategy": "strat2_exp_decay",
        "param": 0.50,
        "C": 3.0,
        "desc": "Strat 2 Continuous Decay: alpha=0.50, C=3.0"
    },
    {
        "id": "Strat2_Exp_a0.75_C2.0",
        "strategy": "strat2_exp_decay",
        "param": 0.75,
        "C": 2.0,
        "desc": "Strat 2 Continuous Decay: alpha=0.75, C=2.0"
    },
    # Strategy 3: Explicit Shortness Feature Block
    {
        "id": "Strat3_Explicit_a0.50_C2.0",
        "strategy": "strat3_explicit_shortness",
        "param": 0.50,
        "C": 2.0,
        "desc": "Strat 3 Explicit Shortness Block: alpha=0.50, C=2.0"
    },
    {
        "id": "Strat3_Explicit_a0.50_C1.0",
        "strategy": "strat3_explicit_shortness",
        "param": 0.50,
        "C": 1.0,
        "desc": "Strat 3 Explicit Shortness Block: alpha=0.50, C=1.0"
    }
]

experiment_results = []
trained_models = {}

for cand in candidate_configs:
    cand_id = cand["id"]
    strat = cand["strategy"]
    param = cand["param"]
    c_val = cand["C"]
    
    # 1. Build interaction block for train & calibration
    X_train_inter = build_interaction_block(strat, param, X_train_embedding, model_train_df, train_word_counts)
    X_cal_inter = build_interaction_block(strat, param, X_calibration_embedding, calibration_df, cal_word_counts)
    
    # 2. Assemble feature matrices
    X_train_cand = hstack([X_train_word, X_train_char, X_train_emb_sparse, X_train_inter, X_train_ling_sparse], format="csr")
    X_cal_cand = hstack([X_calibration_word, X_calibration_char, X_cal_emb_sparse, X_cal_inter, X_cal_ling_sparse], format="csr")
    
    # 3. Train classifier on model_train_df
    clf = LogisticRegression(
        max_iter=2500,
        C=c_val,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
    clf.fit(X_train_cand, y_train, sample_weight=train_sample_weights)
    
    # 4. Predict on calibration set
    cal_probs = clf.predict_proba(X_cal_cand)[:, 1]
    
    # 5. Calibrate length-aware thresholds ONLY on calibration set
    thresholds, sweeps = find_stable_length_thresholds(y_calibration, cal_probs, calibration_df)
    
    # Apply calibrated thresholds to calibration set
    cal_preds = np.array([
        int(cal_probs[i] >= thresholds[calibration_df.iloc[i]["length_group"]])
        for i in range(len(calibration_df))
    ])
    
    cal_acc = accuracy_score(y_calibration, cal_preds)
    cal_prec = precision_score(y_calibration, cal_preds, zero_division=0)
    cal_rec = recall_score(y_calibration, cal_preds, zero_division=0)
    cal_f1 = f1_score(y_calibration, cal_preds, zero_division=0)
    cal_auc = roc_auc_score(y_calibration, cal_probs)
    
    cal_fp = int(((y_calibration == 0) & (cal_preds == 1)).sum())
    cal_fn = int(((y_calibration == 1) & (cal_preds == 0)).sum())
    
    # Short-text calibration F1
    st_mask = calibration_df["length_group"].isin(["very_short", "short"]).values
    st_f1 = f1_score(y_calibration[st_mask], cal_preds[st_mask], zero_division=0)
    
    experiment_results.append({
        "id": cand_id,
        "strategy": strat,
        "C": c_val,
        "cal_accuracy": round(cal_acc, 4),
        "cal_precision": round(cal_prec, 4),
        "cal_recall": round(cal_rec, 4),
        "cal_f1": round(cal_f1, 4),
        "cal_short_f1": round(st_f1, 4),
        "cal_roc_auc": round(cal_auc, 4),
        "cal_FP": cal_fp,
        "cal_FN": cal_fn,
        "thresholds": thresholds,
        "desc": cand["desc"]
    })
    
    trained_models[cand_id] = {
        "clf": clf,
        "thresholds": thresholds,
        "config": cand,
        "sweeps": sweeps
    }
    
    print(f"[{cand_id:26s}] Cal F1={cal_f1:.4f} | Short F1={st_f1:.4f} | Cal Acc={cal_acc:.4f} | Prec={cal_prec:.4f} | Rec={cal_rec:.4f} | FP={cal_fp:2d} | FN={cal_fn:2d} | Thresh={thresholds}")

exp_df = pd.DataFrame(experiment_results)
exp_df.to_csv(os.path.join(V32_MODEL_DIR, "candidate_calibration_experiments.csv"), index=False)

# =========================================================
# SELECT BEST CANDIDATE (STRICTLY FROM CALIBRATION)
# =========================================================

exp_df_sorted = exp_df.sort_values(by=["cal_f1", "cal_short_f1", "cal_precision"], ascending=False)
best_cand_row = exp_df_sorted.iloc[0]
best_cand_id = best_cand_row["id"]
best_cand_data = trained_models[best_cand_id]

print("\n" + "=" * 70)
print(f"SELECTED BEST CANDIDATE BASED ON CALIBRATION: {best_cand_id}")
print(f"Description: {best_cand_data['config']['desc']}")
print(f"Calibration F1: {best_cand_row['cal_f1']:.4f} | Short F1: {best_cand_row['cal_short_f1']:.4f} | Accuracy: {best_cand_row['cal_accuracy']:.4f}")
print(f"Frozen Calibration Thresholds: {best_cand_data['thresholds']}")
print("=" * 70)

# =========================================================
# FINAL HELD-OUT VALIDATION (EVALUATED EXACTLY ONCE)
# =========================================================

print("\n4. FINAL HELD-OUT VALIDATION (UNTOUCHED 998 SAMPLES)")
print("---------------------------------------------------")

winning_config = best_cand_data["config"]
winning_clf = best_cand_data["clf"]
frozen_thresholds = best_cand_data["thresholds"]

X_val_inter_best = build_interaction_block(
    winning_config["strategy"],
    winning_config["param"],
    X_validation_embedding,
    validation_df,
    val_word_counts
)
X_val_best = hstack([X_validation_word, X_validation_char, X_val_emb_sparse, X_val_inter_best, X_val_ling_sparse], format="csr")

val_probs = winning_clf.predict_proba(X_val_best)[:, 1]

applied_val_thresholds = np.array([
    frozen_thresholds[validation_df.iloc[i]["length_group"]]
    for i in range(len(validation_df))
])

val_preds = (val_probs >= applied_val_thresholds).astype(int)

v32_acc = accuracy_score(y_validation, val_preds)
v32_prec = precision_score(y_validation, val_preds, zero_division=0)
v32_rec = recall_score(y_validation, val_preds, zero_division=0)
v32_f1 = f1_score(y_validation, val_preds, zero_division=0)
v32_auc = roc_auc_score(y_validation, val_probs)

cm_v32 = confusion_matrix(y_validation, val_preds)
tn_32, fp_32, fn_32, tp_32 = cm_v32.ravel()

# Build validation results dataframe
validation_result_df = validation_df.copy()
validation_result_df["probability"] = val_probs
validation_result_df["applied_threshold"] = applied_val_thresholds
validation_result_df["prediction"] = val_preds
validation_result_df["error_type"] = "Correct"
validation_result_df.loc[(validation_result_df["label"] == 0) & (validation_result_df["prediction"] == 1), "error_type"] = "False Positive"
validation_result_df.loc[(validation_result_df["label"] == 1) & (validation_result_df["prediction"] == 0), "error_type"] = "False Negative"

validation_result_df.to_csv(os.path.join(V32_MODEL_DIR, "validation_results.csv"), index=False)

# Validation by length group
length_rows = []
for group in ["very_short", "short", "medium", "long"]:
    sub = validation_result_df[validation_result_df["length_group"] == group]
    g_acc = accuracy_score(sub["label"], sub["prediction"])
    g_prec = precision_score(sub["label"], sub["prediction"], zero_division=0)
    g_rec = recall_score(sub["label"], sub["prediction"], zero_division=0)
    g_f1 = f1_score(sub["label"], sub["prediction"], zero_division=0)
    g_fp = ((sub["label"] == 0) & (sub["prediction"] == 1)).sum()
    g_fn = ((sub["label"] == 1) & (sub["prediction"] == 0)).sum()
    
    length_rows.append({
        "length_group": group,
        "samples": len(sub),
        "threshold": frozen_thresholds[group],
        "accuracy": round(g_acc, 4),
        "precision": round(g_prec, 4),
        "recall": round(g_rec, 4),
        "f1": round(g_f1, 4),
        "false_positive": int(g_fp),
        "false_negative": int(g_fn)
    })

length_results_df = pd.DataFrame(length_rows)
length_results_df.to_csv(os.path.join(V32_MODEL_DIR, "length_results.csv"), index=False)

# Save Threshold sweeps
pd.DataFrame(best_cand_data["sweeps"]).to_csv(os.path.join(V32_MODEL_DIR, "threshold_results.csv"), index=False)

# =========================================================
# V3.1 BASELINE COMPARISON
# =========================================================

v31_metrics = {
    "accuracy": 0.8717,
    "precision": 0.8355,
    "recall": 0.9151,
    "f1": 0.8735,
    "roc_auc": 0.9521,
    "FP": 87,
    "FN": 41,
    "very_short_f1": 0.8443,
    "short_f1": 0.8598,
    "medium_f1": 0.9185,
    "long_f1": 0.9000
}

comparison_rows = [
    {
        "metric": "Accuracy",
        "V3.1": v31_metrics["accuracy"],
        "V3.2": round(v32_acc, 4),
        "diff": round(v32_acc - v31_metrics["accuracy"], 4)
    },
    {
        "metric": "Precision",
        "V3.1": v31_metrics["precision"],
        "V3.2": round(v32_prec, 4),
        "diff": round(v32_prec - v31_metrics["precision"], 4)
    },
    {
        "metric": "Recall",
        "V3.1": v31_metrics["recall"],
        "V3.2": round(v32_rec, 4),
        "diff": round(v32_rec - v31_metrics["recall"], 4)
    },
    {
        "metric": "F1 Score",
        "V3.1": v31_metrics["f1"],
        "V3.2": round(v32_f1, 4),
        "diff": round(v32_f1 - v31_metrics["f1"], 4)
    },
    {
        "metric": "ROC-AUC",
        "V3.1": v31_metrics["roc_auc"],
        "V3.2": round(v32_auc, 4),
        "diff": round(v32_auc - v31_metrics["roc_auc"], 4)
    },
    {
        "metric": "False Positives (FP)",
        "V3.1": v31_metrics["FP"],
        "V3.2": int(fp_32),
        "diff": int(fp_32 - v31_metrics["FP"])
    },
    {
        "metric": "False Negatives (FN)",
        "V3.1": v31_metrics["FN"],
        "V3.2": int(fn_32),
        "diff": int(fn_32 - v31_metrics["FN"])
    }
]

df_comp = pd.DataFrame(comparison_rows)
df_comp.to_csv(os.path.join(V32_MODEL_DIR, "model_comparison.csv"), index=False)

# =========================================================
# SAVE V3.2 ARTIFACTS
# =========================================================

print("\n5. SAVING V3.2 ARTIFACTS")
print("-----------------------")

save_pickle(word_vectorizer, os.path.join(V32_MODEL_DIR, "word_tfidf_vectorizer.pkl"))
save_pickle(char_vectorizer, os.path.join(V32_MODEL_DIR, "char_tfidf_vectorizer.pkl"))
save_pickle(linguistic_scaler, os.path.join(V32_MODEL_DIR, "linguistic_scaler.pkl"))
save_pickle(winning_clf, os.path.join(V32_MODEL_DIR, "text_classifier.pkl"))

config_v32 = {
    "transformer": TRANSFORMER_NAME,
    "word_max_features": WORD_MAX_FEATURES,
    "char_max_features": CHAR_MAX_FEATURES,
    "linguistic_features": ["log_word_count", "sentence_count", "avg_sentence_length", "vocabulary_diversity", "length_bucket"],
    "interaction_strategy": winning_config["strategy"],
    "interaction_param": winning_config["param"],
    "classifier_C": winning_config["C"],
    "random_seed": RANDOM_SEED,
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df),
    "training_datasets": ["HC3", "RAID", "SentenceAI"],
    "decision_threshold": frozen_thresholds,
    "length_thresholds": frozen_thresholds,
    "length_targets": TARGETS,
    "model_version": "text_final_v3.2"
}
save_pickle(config_v32, os.path.join(V32_MODEL_DIR, "text_config.pkl"))

metrics_v32 = {
    "accuracy": float(v32_acc),
    "precision": float(v32_prec),
    "recall": float(v32_rec),
    "f1": float(v32_f1),
    "roc_auc": float(v32_auc),
    "decision_threshold": frozen_thresholds,
    "length_thresholds": frozen_thresholds,
    "true_negative": int(tn_32),
    "false_positive": int(fp_32),
    "false_negative": int(fn_32),
    "true_positive": int(tp_32),
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df),
    "model_version": "text_final_v3.2"
}
save_pickle(metrics_v32, os.path.join(V32_MODEL_DIR, "text_metrics.pkl"))

# Save Model Summary CSV
summary_v32 = {
    "model": "text_final_v3.2",
    "strategy": winning_config["strategy"],
    "C": winning_config["C"],
    "accuracy": round(v32_acc, 4),
    "precision": round(v32_prec, 4),
    "recall": round(v32_rec, 4),
    "f1": round(v32_f1, 4),
    "roc_auc": round(v32_auc, 4),
    "thresholds": str(frozen_thresholds),
    "false_positive": int(fp_32),
    "false_negative": int(fn_32),
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df)
}
pd.DataFrame([summary_v32]).to_csv(os.path.join(V32_MODEL_DIR, "model_summary.csv"), index=False)

# =========================================================
# ERROR ANALYSIS FOR V3.2
# =========================================================

error_columns = ["text", "label", "probability", "length_group", "applied_threshold", "dataset", "prediction", "error_type"]
fn_32_df = validation_result_df[validation_result_df["error_type"] == "False Negative"][error_columns].copy()
fp_32_df = validation_result_df[validation_result_df["error_type"] == "False Positive"][error_columns].copy()
all_err_32_df = validation_result_df[validation_result_df["error_type"].isin(["False Negative", "False Positive"])][error_columns].copy()

fn_32_df.to_csv(os.path.join(V32_ERROR_DIR, "false_negatives.csv"), index=False)
fp_32_df.to_csv(os.path.join(V32_ERROR_DIR, "false_positives.csv"), index=False)
all_err_32_df.to_csv(os.path.join(V32_ERROR_DIR, "all_errors.csv"), index=False)

# Error analysis by length
length_err_rows = []
for grp in ["very_short", "short", "medium", "long"]:
    sub = validation_result_df[validation_result_df["length_group"] == grp]
    sub_human = (sub["label"] == 0).sum()
    sub_ai = (sub["label"] == 1).sum()
    grp_fp = ((sub["label"] == 0) & (sub["prediction"] == 1)).sum()
    grp_fn = ((sub["label"] == 1) & (sub["prediction"] == 0)).sum()
    length_err_rows.append({
        "length_group": grp,
        "total_samples": len(sub),
        "FP": int(grp_fp),
        "FN": int(grp_fn),
        "FP_rate": round(grp_fp / sub_human if sub_human > 0 else 0, 4),
        "FN_rate": round(grp_fn / sub_ai if sub_ai > 0 else 0, 4)
    })
pd.DataFrame(length_err_rows).to_csv(os.path.join(V32_ERROR_DIR, "error_analysis_by_length.csv"), index=False)

# Ranked error files
hardest_fn_32 = fn_32_df.sort_values(by="probability", ascending=True)
hardest_fp_32 = fp_32_df.sort_values(by="probability", ascending=False)
hardest_fn_32.to_csv(os.path.join(V32_ERROR_DIR, "hardest_false_negatives.csv"), index=False)
hardest_fp_32.to_csv(os.path.join(V32_ERROR_DIR, "hardest_false_positives.csv"), index=False)

# Write V3.2 Error Analysis Report
report_32 = f"""================================================================================
V3.2 ERROR ANALYSIS & EXPERIMENT REPORT
================================================================================
Model Version: text_final_v3.2
Selected Winning Strategy: {winning_config['id']} ({winning_config['desc']})
Decision Thresholds (Frozen from Calibration):
  very_short: {frozen_thresholds['very_short']:.2f}
  short     : {frozen_thresholds['short']:.2f}
  medium    : {frozen_thresholds['medium']:.2f}
  long      : {frozen_thresholds['long']:.2f}

================================================================================
1. HELD-OUT VALIDATION RESULTS (UNTOUCHED 998 SAMPLES)
================================================================================
Accuracy : {v32_acc:.4f}
Precision: {v32_prec:.4f}
Recall   : {v32_rec:.4f}
F1 Score : {v32_f1:.4f}
ROC-AUC  : {v32_auc:.4f}
Confusion Matrix:
  TN: {tn_32} | FP: {fp_32}
  FN: {fn_32} | TP: {tp_32}

================================================================================
2. PERFORMANCE BY LENGTH GROUP
================================================================================
{length_results_df.to_string(index=False)}

================================================================================
3. COMPARISON WITH V3.1 BASELINE
================================================================================
{df_comp.to_string(index=False)}
"""

with open(os.path.join(V32_ERROR_DIR, "V3.2_ERROR_ANALYSIS_REPORT.txt"), "w", encoding="utf-8") as f:
    f.write(report_32)

print("\n" + "=" * 70)
print("EXPERIMENT EXECUTION COMPLETE")
print("=" * 70)
