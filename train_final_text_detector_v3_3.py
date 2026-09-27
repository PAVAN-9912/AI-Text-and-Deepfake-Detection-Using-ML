import os
import re
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
print("V3.3 GENERALIZED AI TEXT DETECTOR TRAINING")
print("==========================================")
print("Base Architecture: HC3 + RAID + SentenceAI")
print("Features: Word TF-IDF + Char TF-IDF + MiniLM + Short Interaction + Style & Code Features")
print("==========================================")

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

V32_MODEL_DIR = r"models\text_final_v3_2"
V33_MODEL_DIR = r"models\text_final_v3_3"
V33_ERROR_DIR = os.path.join(V33_MODEL_DIR, "error_analysis")

os.makedirs(V33_MODEL_DIR, exist_ok=True)
os.makedirs(V33_ERROR_DIR, exist_ok=True)

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
ALPHA_SHORT_DECAY = 0.75
CLASSIFIER_C = 2.0

# =========================================================
# REGEX PATTERNS FOR STYLE & CODE FEATURES
# =========================================================

contractions = [
    r"can't", r"won't", r"don't", r"doesn't", r"didn't", r"isn't", r"aren't",
    r"wasn't", r"weren't", r"haven't", r"hasn't", r"hadn't", r"i'm", r"i've",
    r"i'll", r"i'd", r"you're", r"you've", r"you'll", r"you'd", r"he's",
    r"she's", r"it's", r"that's", r"we're", r"we've", r"we'll", r"they'd",
    r"they're", r"they've", r"they'll", r"couldn't", r"shouldn't", r"wouldn't",
    r"there's", r"what's", r"let's", r"who's"
]
contraction_pattern = r"\b(" + "|".join(c.replace("'", r"['\u2019]") for c in contractions) + r")\b"
contraction_regex = re.compile(contraction_pattern, re.IGNORECASE)

pronouns = [
    r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b",
    r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b",
    r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b", r"\byourselves\b"
]
pronoun_regex = re.compile("|".join(pronouns), re.IGNORECASE)

code_kw_regex = re.compile(
    r"(\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:\(]|\breturn\b|\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import|\bwhile\s+.*:|\bfor\s+\w+\s+in\s+|\belif\s+.*:|\bexcept\s*.*:|\blambda\s+.*:|\bprint\s*\(|\bself\.\w+)",
    re.IGNORECASE
)
code_sym_regex = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\b\d+\s*//\s*\d+\b|->|::|\{|\}|\[\s*\]|\(\s*\)|;\s*$)",
    re.MULTILINE
)

FEATURE_NAMES = [
    # 5 base linguistic
    "log_word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity",
    "length_bucket",
    # 7 style features
    "contraction_density",
    "pronoun_density",
    "question_mark_density",
    "exclamation_density",
    "ellipsis_density",
    "uppercase_word_ratio",
    "irregular_caps_ratio",
    # 5 code features
    "code_keyword_density",
    "code_symbol_density",
    "function_def_flag",
    "import_stmt_flag",
    "code_likelihood"
]

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

def extract_all_linguistic_style_code_features(texts):
    features = []
    for text in texts:
        text = str(text)
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        wc_safe = max(word_count, 1)

        # 1. Base linguistic features
        sentence_count = sum(1 for char in text if char in ".!?")
        if sentence_count == 0:
            sentence_count = 1

        avg_sentence_length = word_count / max(sentence_count, 1)

        if word_count > 0:
            unique_words = len(set(w.lower() for w in words))
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

        # 2. Style features
        contraction_count = len(contraction_regex.findall(text))
        contraction_density = contraction_count / wc_safe

        pronoun_count = len(pronoun_regex.findall(text))
        pronoun_density = pronoun_count / wc_safe

        question_mark_density = text.count('?') / wc_safe
        exclamation_density = len(re.findall(r'!(?!=)', text)) / wc_safe
        ellipsis_density = len(re.findall(r'\.{2,}', text)) / wc_safe

        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        uppercase_word_ratio = upper_words / wc_safe

        irreg_caps = sum(1 for w in words if any(c.isupper() for c in w[1:]) and not w.isupper())
        irregular_caps_ratio = irreg_caps / wc_safe

        # 3. Code / Syntax features
        code_kw_count = len(code_kw_regex.findall(text))
        code_keyword_density = code_kw_count / wc_safe

        code_sym_count = len(code_sym_regex.findall(text))
        code_symbol_density = code_sym_count / wc_safe

        function_def_flag = 1.0 if bool(re.search(r'\bdef\s+[a-zA-Z_]\w*\s*\(', text)) else 0.0
        import_stmt_flag = 1.0 if bool(re.search(r'(\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import)', text)) else 0.0
        has_indentation = 1.0 if bool(re.search(r'^\s{2,}\S', text, re.MULTILINE)) else 0.0

        code_likelihood = min(1.0, (code_keyword_density * 2.5 + code_symbol_density * 2.0 + function_def_flag * 0.4 + import_stmt_flag * 0.3 + has_indentation * 0.2))

        features.append([
            log_word_count,
            sentence_count,
            avg_sentence_length,
            vocabulary_diversity,
            length_bucket,
            contraction_density,
            pronoun_density,
            question_mark_density,
            exclamation_density,
            ellipsis_density,
            uppercase_word_ratio,
            irregular_caps_ratio,
            code_keyword_density,
            code_symbol_density,
            function_def_flag,
            import_stmt_flag,
            code_likelihood
        ])

    return np.asarray(features, dtype=np.float32)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# =========================================================
# DATASET LOADING (EXACT REPLICATION OF V3/V3.1/V3.2)
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

# Balanced sampling
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

# Save Training Distribution
distribution = pd.crosstab([train_df["dataset"], train_df["length_group"]], train_df["label"]).reset_index()
distribution.to_csv(os.path.join(V33_MODEL_DIR, "training_distribution.csv"), index=False)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

print("\n2. EXTRACTING FEATURES")
print("----------------------")

# 1. Word TF-IDF
print("Fitting Word TF-IDF (150k)...")
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

# 2. Char TF-IDF
print("Fitting Character TF-IDF (150k)...")
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

# 3. MiniLM Embeddings
print("Encoding MiniLM Embeddings...")
transformer_model = SentenceTransformer(TRANSFORMER_NAME)
X_train_embedding = transformer_model.encode(model_train_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_calibration_embedding = transformer_model.encode(calibration_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_validation_embedding = transformer_model.encode(validation_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

# 4. Short-Text Interaction (Exponential Decay alpha=0.75)
train_word_counts = np.array([len(t.split()) for t in model_train_df["text"]], dtype=np.float32)
cal_word_counts = np.array([len(t.split()) for t in calibration_df["text"]], dtype=np.float32)
val_word_counts = np.array([len(t.split()) for t in validation_df["text"]], dtype=np.float32)

decay_train = (ALPHA_SHORT_DECAY * np.exp(-train_word_counts / 15.0)).reshape(-1, 1).astype(np.float32)
decay_cal = (ALPHA_SHORT_DECAY * np.exp(-cal_word_counts / 15.0)).reshape(-1, 1).astype(np.float32)
decay_val = (ALPHA_SHORT_DECAY * np.exp(-val_word_counts / 15.0)).reshape(-1, 1).astype(np.float32)

X_train_short_inter = csr_matrix((X_train_embedding * decay_train).astype(np.float32))
X_cal_short_inter = csr_matrix((X_calibration_embedding * decay_cal).astype(np.float32))
X_val_short_inter = csr_matrix((X_validation_embedding * decay_val).astype(np.float32))

# 5. Expanded 17 Linguistic, Style & Code Features
print("Building Expanded Style & Code Features (17 total)...")
X_train_style_raw = extract_all_linguistic_style_code_features(model_train_df["text"].tolist())
X_cal_style_raw = extract_all_linguistic_style_code_features(calibration_df["text"].tolist())
X_val_style_raw = extract_all_linguistic_style_code_features(validation_df["text"].tolist())

linguistic_scaler = StandardScaler()
X_train_style = linguistic_scaler.fit_transform(X_train_style_raw)
X_calibration_style = linguistic_scaler.transform(X_cal_style_raw)
X_validation_style = linguistic_scaler.transform(X_val_style_raw)

# Assemble feature matrices
X_train = hstack([
    X_train_word,
    X_train_char,
    csr_matrix(X_train_embedding),
    X_train_short_inter,
    csr_matrix(X_train_style)
], format="csr")

X_calibration = hstack([
    X_calibration_word,
    X_calibration_char,
    csr_matrix(X_calibration_embedding),
    X_cal_short_inter,
    csr_matrix(X_calibration_style)
], format="csr")

X_validation = hstack([
    X_validation_word,
    X_validation_char,
    csr_matrix(X_validation_embedding),
    X_val_short_inter,
    csr_matrix(X_validation_style)
], format="csr")

print("Feature Matrix Shapes:")
print(f"  Word TF-IDF:          {X_train_word.shape}")
print(f"  Char TF-IDF:          {X_train_char.shape}")
print(f"  MiniLM Embeddings:    {X_train_embedding.shape}")
print(f"  Short Interaction:    {X_train_short_inter.shape}")
print(f"  Style & Code Features:{X_train_style.shape}")
print(f"  Combined Matrix:      {X_train.shape}")

# Labels & Sample weights
y_train = model_train_df["label"].values.astype(int)
y_calibration = calibration_df["label"].values.astype(int)
y_validation = validation_df["label"].values.astype(int)

length_weights = {"very_short": 1.25, "short": 1.15, "medium": 1.05, "long": 1.00}
train_sample_weights = model_train_df["length_group"].map(length_weights).astype(float).values

# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print("\n3. TRAINING V3.3 LOGISTIC REGRESSION CLASSIFIER")
print("-----------------------------------------------")

classifier = LogisticRegression(
    max_iter=2500,
    C=CLASSIFIER_C,
    solver="liblinear",
    class_weight="balanced",
    random_state=RANDOM_SEED
)
classifier.fit(X_train, y_train, sample_weight=train_sample_weights)
print("Classifier training completed.")

# =========================================================
# CALIBRATE LENGTH-AWARE THRESHOLDS (CALIBRATION ONLY)
# =========================================================

print("\n4. CALIBRATING LENGTH-AWARE THRESHOLDS (CALIBRATION SET ONLY)")
print("-------------------------------------------------------------")

cal_probabilities = classifier.predict_proba(X_calibration)[:, 1]

length_thresholds = {}
threshold_results_rows = []

for group in ["very_short", "short", "medium", "long"]:
    mask = (calibration_df["length_group"] == group).values
    y_g = y_calibration[mask]
    p_g = cal_probabilities[mask]
    support = len(y_g)

    group_sweeps = []
    for t in np.arange(0.20, 0.81, 0.01):
        t_val = round(float(t), 2)
        preds = (p_g >= t_val).astype(int)
        f1 = f1_score(y_g, preds, zero_division=0)
        acc = accuracy_score(y_g, preds)
        prec = precision_score(y_g, preds, zero_division=0)
        rec = recall_score(y_g, preds, zero_division=0)
        fp_count = int(((y_g == 0) & (preds == 1)).sum())
        fn_count = int(((y_g == 1) & (preds == 0)).sum())

        sweep_row = {
            "length_group": group,
            "threshold": t_val,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "FP": fp_count,
            "FN": fn_count
        }
        group_sweeps.append(sweep_row)
        threshold_results_rows.append(sweep_row)

    df_sw = pd.DataFrame(group_sweeps)
    max_f1 = df_sw["f1"].max()

    # Stable plateau selection (near-optimal within 0.005)
    near_optimal = df_sw[df_sw["f1"] >= (max_f1 - 0.005)]
    chosen_thresh = float(near_optimal["threshold"].median())
    chosen_thresh = round(chosen_thresh, 2)
    length_thresholds[group] = chosen_thresh

    best_match = df_sw[df_sw["threshold"] == chosen_thresh].iloc[0]
    print(f"Group: {group:12s} | N={support:3d} | Selected Threshold: {chosen_thresh:.2f} | Cal F1: {best_match['f1']:.4f} | Acc: {best_match['accuracy']:.4f} | Prec: {best_match['precision']:.4f} | Rec: {best_match['recall']:.4f} | FP: {best_match['FP']:2d} | FN: {best_match['FN']:2d}")

pd.DataFrame(threshold_results_rows).to_csv(os.path.join(V33_MODEL_DIR, "threshold_results.csv"), index=False)

print("\nFROZEN LENGTH-AWARE THRESHOLDS FOR V3.3:")
for grp, thresh in length_thresholds.items():
    print(f" - {grp:12s}: {thresh:.2f}")

# =========================================================
# FINAL HELD-OUT VALIDATION (UNTOUCHED 998 SAMPLES)
# =========================================================

print("\n5. FINAL HELD-OUT VALIDATION (EVALUATED EXACTLY ONCE)")
print("-----------------------------------------------------")

validation_probabilities = classifier.predict_proba(X_validation)[:, 1]

applied_thresholds = np.array([
    length_thresholds[validation_df.iloc[i]["length_group"]]
    for i in range(len(validation_df))
])

validation_predictions = (validation_probabilities >= applied_thresholds).astype(int)

v33_acc = accuracy_score(y_validation, validation_predictions)
v33_prec = precision_score(y_validation, validation_predictions, zero_division=0)
v33_rec = recall_score(y_validation, validation_predictions, zero_division=0)
v33_f1 = f1_score(y_validation, validation_predictions, zero_division=0)
v33_auc = roc_auc_score(y_validation, validation_probabilities)

cm = confusion_matrix(y_validation, validation_predictions)
tn, fp, fn, tp = cm.ravel()

human_total = (y_validation == 0).sum()
ai_total = (y_validation == 1).sum()
fp_rate = fp / human_total
fn_rate = fn / ai_total

print("=" * 70)
print("V3.3 FINAL VALIDATION METRICS")
print("=" * 70)
print(f"Accuracy : {v33_acc:.4f} ({tn+tp}/{len(y_validation)})")
print(f"Precision: {v33_prec:.4f}")
print(f"Recall   : {v33_rec:.4f}")
print(f"F1 Score : {v33_f1:.4f}")
print(f"ROC-AUC  : {v33_auc:.4f}")
print(f"TN: {tn} | FP: {fp} (FP Rate: {fp_rate:.4f})")
print(f"FN: {fn} | TP: {tp} (FN Rate: {fn_rate:.4f})")

# Build Validation Results DataFrame
validation_result_df = validation_df.copy()
validation_result_df["probability"] = validation_probabilities
validation_result_df["applied_threshold"] = applied_thresholds
validation_result_df["prediction"] = validation_predictions
validation_result_df["error_type"] = "Correct"
validation_result_df.loc[(validation_result_df["label"] == 0) & (validation_result_df["prediction"] == 1), "error_type"] = "False Positive"
validation_result_df.loc[(validation_result_df["label"] == 1) & (validation_result_df["prediction"] == 0), "error_type"] = "False Negative"

# Add extra diagnostic feature columns for error analysis
val_extra_feats = X_val_style_raw
validation_result_df["contraction_density"] = val_extra_feats[:, 5]
validation_result_df["pronoun_density"] = val_extra_feats[:, 6]
validation_result_df["question_density"] = val_extra_feats[:, 7]
validation_result_df["exclamation_density"] = val_extra_feats[:, 8]
validation_result_df["capitalization_irregularity"] = val_extra_feats[:, 11]
validation_result_df["code_likelihood"] = val_extra_feats[:, 16]

validation_result_df.to_csv(os.path.join(V33_MODEL_DIR, "validation_results.csv"), index=False)

# Validation by length group
length_rows = []
for group in ["very_short", "short", "medium", "long"]:
    sub = validation_result_df[validation_result_df["length_group"] == group]
    g_acc = accuracy_score(sub["label"], sub["prediction"])
    g_prec = precision_score(sub["label"], sub["prediction"], zero_division=0)
    g_rec = recall_score(sub["label"], sub["prediction"], zero_division=0)
    g_f1 = f1_score(sub["label"], sub["prediction"], zero_division=0)
    g_fp = int(((sub["label"] == 0) & (sub["prediction"] == 1)).sum())
    g_fn = int(((sub["label"] == 1) & (sub["prediction"] == 0)).sum())
    g_human = int((sub["label"] == 0).sum())
    g_ai = int((sub["label"] == 1).sum())

    length_rows.append({
        "length_group": group,
        "samples": len(sub),
        "threshold": length_thresholds[group],
        "accuracy": round(g_acc, 4),
        "precision": round(g_prec, 4),
        "recall": round(g_rec, 4),
        "f1": round(g_f1, 4),
        "false_positive": g_fp,
        "false_negative": g_fn,
        "FP_rate": round(g_fp / g_human if g_human > 0 else 0, 4),
        "FN_rate": round(g_fn / g_ai if g_ai > 0 else 0, 4)
    })

length_results_df = pd.DataFrame(length_rows)
length_results_df.to_csv(os.path.join(V33_MODEL_DIR, "length_results.csv"), index=False)

print("\nPERFORMANCE BY LENGTH GROUP:")
print(length_results_df.to_string(index=False))

# =========================================================
# V3.2 VS V3.3 ERROR TRANSITION ANALYSIS
# =========================================================

print("\n6. V3.2 VS V3.3 ERROR TRANSITIONS")
print("---------------------------------")

v32_val_file = os.path.join(V32_MODEL_DIR, "validation_results.csv")
if os.path.exists(v32_val_file):
    v32_val = pd.read_csv(v32_val_file)
    v32_preds = v32_val["prediction"].values
    v33_preds = validation_predictions
    y_true = y_validation

    transitions = []
    for i in range(len(y_true)):
        y = y_true[i]
        p32 = v32_preds[i]
        p33 = v33_preds[i]

        if p32 == y and p33 == y:
            cat = "V3.2_correct_to_V3.3_correct"
        elif p32 != y and p33 == y:
            cat = "V3.2_FP_to_V3.3_correct" if p32 == 1 else "V3.2_FN_to_V3.3_correct"
        elif p32 == y and p33 != y:
            cat = "V3.2_correct_to_V3.3_FP" if p33 == 1 else "V3.2_correct_to_V3.3_FN"
        else:
            cat = "persistent_FP" if p33 == 1 else "persistent_FN"
        transitions.append(cat)

    transition_df = validation_result_df.copy()
    transition_df["transition"] = transitions
    transition_df["v32_prediction"] = v32_preds
    transition_df["v32_probability"] = v32_val["probability"]
    transition_df["v32_threshold"] = v32_val["applied_threshold"]
    transition_df["v33_prediction"] = v33_preds
    transition_df["v33_probability"] = validation_probabilities
    transition_df["v33_threshold"] = applied_thresholds

    transition_df.to_csv(os.path.join(V33_ERROR_DIR, "V32_VS_V33_ERROR_TRANSITIONS.csv"), index=False)

    print("Transition Breakdown:")
    print(pd.Series(transitions).value_counts().to_string())

# Save Error Specific CSVs
error_cols = [
    "text", "label", "prediction", "probability", "applied_threshold", "length_group", "dataset", "error_type",
    "code_likelihood", "contraction_density", "pronoun_density", "question_density", "exclamation_density", "capitalization_irregularity"
]
v33_fp_df = validation_result_df[validation_result_df["error_type"] == "False Positive"][error_cols].copy()
v33_fn_df = validation_result_df[validation_result_df["error_type"] == "False Negative"][error_cols].copy()

v33_fp_df.to_csv(os.path.join(V33_ERROR_DIR, "V33_FALSE_POSITIVES.csv"), index=False)
v33_fn_df.to_csv(os.path.join(V33_ERROR_DIR, "V33_FALSE_NEGATIVES.csv"), index=False)

if os.path.exists(v32_val_file):
    persistent_mask = transition_df["transition"].isin(["persistent_FP", "persistent_FN"])
    v33_persist_df = transition_df[persistent_mask][error_cols + ["transition"]].copy()
    v33_persist_df.to_csv(os.path.join(V33_ERROR_DIR, "V33_PERSISTENT_ERRORS.csv"), index=False)

# =========================================================
# SAVE V3.3 ARTIFACTS
# =========================================================

print("\n7. SAVING V3.3 ARTIFACTS")
print("-----------------------")

save_pickle(word_vectorizer, os.path.join(V33_MODEL_DIR, "word_tfidf_vectorizer.pkl"))
save_pickle(char_vectorizer, os.path.join(V33_MODEL_DIR, "char_tfidf_vectorizer.pkl"))
save_pickle(linguistic_scaler, os.path.join(V33_MODEL_DIR, "linguistic_scaler.pkl"))
save_pickle(classifier, os.path.join(V33_MODEL_DIR, "text_classifier.pkl"))

config_v33 = {
    "model_version": "text_final_v3.3",
    "transformer": TRANSFORMER_NAME,
    "word_max_features": WORD_MAX_FEATURES,
    "char_max_features": CHAR_MAX_FEATURES,
    "linguistic_feature_names": FEATURE_NAMES[:5],
    "style_feature_names": FEATURE_NAMES[5:12],
    "code_feature_names": FEATURE_NAMES[12:],
    "all_feature_names": FEATURE_NAMES,
    "total_scalar_features": len(FEATURE_NAMES),
    "short_interaction_alpha": ALPHA_SHORT_DECAY,
    "classifier_C": CLASSIFIER_C,
    "random_seed": RANDOM_SEED,
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df),
    "training_datasets": ["HC3", "RAID", "SentenceAI"],
    "decision_threshold": length_thresholds,
    "length_thresholds": length_thresholds,
    "length_targets": TARGETS,
    "decision_policy": "length_aware_frozen_calibration_thresholds"
}
save_pickle(config_v33, os.path.join(V33_MODEL_DIR, "text_config.pkl"))

metrics_v33 = {
    "accuracy": float(v33_acc),
    "precision": float(v33_prec),
    "recall": float(v33_rec),
    "f1": float(v33_f1),
    "roc_auc": float(v33_auc),
    "decision_threshold": length_thresholds,
    "length_thresholds": length_thresholds,
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "fp_rate": float(fp_rate),
    "fn_rate": float(fn_rate),
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df),
    "model_version": "text_final_v3.3"
}
save_pickle(metrics_v33, os.path.join(V33_MODEL_DIR, "text_metrics.pkl"))

# Save Model Summary CSV
summary_v33 = {
    "model": "text_final_v3.3",
    "accuracy": round(v33_acc, 4),
    "precision": round(v33_prec, 4),
    "recall": round(v33_rec, 4),
    "f1": round(v33_f1, 4),
    "roc_auc": round(v33_auc, 4),
    "thresholds": str(length_thresholds),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "fp_rate": round(fp_rate, 4),
    "fn_rate": round(fn_rate, 4),
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df)
}
pd.DataFrame([summary_v33]).to_csv(os.path.join(V33_MODEL_DIR, "model_summary.csv"), index=False)

print("\nV3.3 TRAINING COMPLETED")
print("Artifact directory:", V33_MODEL_DIR)
