import os
import sys
import math
import pickle
import random
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

EXPERIMENTS_DIR = os.path.join(BASE_DIR, "models", "text_experiments_v3")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("PHASE C & D & E & F — TEXT DETECTOR V3 COMPREHENSIVE EXPERIMENT ENGINE")
print("=" * 80)

# ----------------------------------------------------------------------
# 1. LOAD DATASETS & RECONSTRUCT IDENTICAL 8160/907/998 SPLITS
# ----------------------------------------------------------------------
HC3_TRAIN = os.path.join(BASE_DIR, "datasets", "text", "HC3", "train.csv")
HC3_VALIDATION = os.path.join(BASE_DIR, "datasets", "text", "HC3", "validation.csv")
RAID_TRAIN = os.path.join(BASE_DIR, "datasets", "text", "RAID", "raid_domain_training_sample.csv")
SENTENCE_TRAIN = os.path.join(BASE_DIR, "datasets", "text", "SentenceAI", "sentence_ai_training.csv")

RAID_EXTERNAL = os.path.join(BASE_DIR, "datasets", "text", "RAID", "raid_external.csv")
SENTENCE_TEST = os.path.join(BASE_DIR, "datasets", "text", "SentenceAI", "sentence_ai_test.csv")
REAL_WORLD_TEST = os.path.join(BASE_DIR, "datasets", "text", "realworld_test.csv")
AIGCODESET_TEST = os.path.join(BASE_DIR, "datasets", "text", "CodeAI", "aigcodeset_test.csv")

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

# 10% held-out validation (strictly untouched)
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

strat_key = remaining_train_df["label"].astype(str) + "_" + remaining_train_df["length_group"].astype(str)
core_idx, cal_idx = train_test_split(
    np.arange(len(remaining_train_df)),
    test_size=0.10,
    random_state=42,
    stratify=strat_key
)
model_train_df = remaining_train_df.iloc[core_idx].reset_index(drop=True)
calibration_df = remaining_train_df.iloc[cal_idx].reset_index(drop=True)

print(f"Data Splits: Train={len(model_train_df)}, Calibration={len(calibration_df)}, Held-out Validation={len(validation_df)}")

# Load External Benchmark Sets
ext_dfs = {}
if os.path.exists(RAID_EXTERNAL):
    rdf = clean_dataframe(pd.read_csv(RAID_EXTERNAL))
    if "label" in rdf.columns:
        rdf["length_group"] = rdf["text"].apply(get_length_group)
        ext_dfs["RAID_External"] = rdf

if os.path.exists(SENTENCE_TEST):
    sdf = clean_dataframe(pd.read_csv(SENTENCE_TEST))
    if "label_name" in sdf.columns and "label" not in sdf.columns:
        sdf["label"] = sdf["label_name"].map({"Human": 0, "AI": 1})
    sdf = sdf[sdf["label"].isin([0, 1])].copy()
    sdf["length_group"] = sdf["text"].apply(get_length_group)
    ext_dfs["SentenceAI_Test"] = sdf

if os.path.exists(REAL_WORLD_TEST):
    rwdf = clean_dataframe(pd.read_csv(REAL_WORLD_TEST))
    if "label" in rwdf.columns:
        rwdf["length_group"] = rwdf["text"].apply(get_length_group)
        ext_dfs["Real_World_Test"] = rwdf

if os.path.exists(AIGCODESET_TEST):
    cdf = clean_dataframe(pd.read_csv(AIGCODESET_TEST))
    if "label" in cdf.columns:
        cdf["length_group"] = cdf["text"].apply(get_length_group)
        ext_dfs["AIGCodeSet_Test"] = cdf

# ----------------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE
# ----------------------------------------------------------------------
contractions = [
    r"can't", r"cannot", r"won't", r"don't", r"doesn't", r"didn't", r"isn't", r"aren't",
    r"wasn't", r"weren't", r"haven't", r"hasn't", r"hadn't", r"i'm", r"i've",
    r"i'll", r"i'd", r"you're", r"you've", r"you'll", r"you'd", r"he's",
    r"she's", r"it's", r"that's", r"we're", r"we've", r"we'll", r"they'd",
    r"they're", r"they've", r"they'll", r"couldn't", r"shouldn't", r"wouldn't",
    r"there's", r"what's", r"let's", r"who's"
]
contraction_regex = re.compile(r"\b(" + "|".join(c.replace("'", r"['\u2019]") for c in contractions) + r")\b", re.IGNORECASE)

pronouns = [
    r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b",
    r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b",
    r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b", r"\byourselves\b",
    r"\bthey\b", r"\bthem\b", r"\btheir\b", r"\btheirs\b", r"\bthemselves\b"
]
pronoun_regex = re.compile("|".join(pronouns), re.IGNORECASE)

transition_words = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b", r"\bin addition\b",
    r"\bhowever\b", r"\btherefore\b", r"\bthus\b", r"\bconsequently\b",
    r"\bfirstly\b", r"\bsecondly\b", r"\bfinally\b", r"\bto summarize\b",
    r"\bon the other hand\b", r"\bas a result\b", r"\bin contrast\b"
]
transition_regex = re.compile("|".join(transition_words), re.IGNORECASE)

code_kw_regex = re.compile(
    r"(\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:\(]|\breturn\b|\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import|\bwhile\s+.*:|\bfor\s+\w+\s+in\s+|\belif\s+.*:|\belse\s*:|\btry\s*:|\bexcept\s*.*:|\blambda\b|\byield\b|\bprint\s*\(|\bself\.\w+)",
    re.IGNORECASE
)
code_sym_regex = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\*\*|\b\d+\s*//\s*\d+\b|->|::|\{|\}|\[\s*\]|\(\s*\)|;\s*$)",
    re.MULTILINE
)
german_chars_regex = re.compile(r"[äöüÄÖÜß]")
accented_chars_regex = re.compile(r"[éèêëáàâäãåíìîïóòôöõúùûüñç]", re.IGNORECASE)

def calculate_token_entropy(words):
    if not words: return 0.0
    lengths = [len(w) for w in words]
    counts = Counter(lengths)
    total = len(lengths)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))

def calculate_rep_ngrams(words, n=3):
    if len(words) < n: return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return float((len(ngrams) - len(set(ngrams))) / max(len(ngrams), 1))

def calculate_sent_len_var(text, words):
    raw_s = [s.strip().split() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(raw_s) <= 1: return 0.0
    lens = [len(s) for s in raw_s]
    return float(np.std(lens) / max(np.mean(lens), 1.0))

def extract_base_scalar_features(texts):
    features = []
    for text in texts:
        text = str(text).strip()
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        wc_safe = max(word_count, 1)
        cc_safe = max(char_count, 1)

        sentence_count = max(sum(1 for c in text if c in ".!?"), 1)
        avg_sentence_length = word_count / sentence_count
        vocab_diversity = len(set(w.lower() for w in words)) / wc_safe if word_count > 0 else 0.0
        log_word_count = np.log1p(word_count)

        if word_count <= 10: length_bucket = 0.0
        elif word_count <= 30: length_bucket = 1.0
        elif word_count <= 60: length_bucket = 2.0
        else: length_bucket = 3.0

        token_entropy = calculate_token_entropy(words)
        rep_ngram_ratio = calculate_rep_ngrams(words, n=3)
        transition_ratio = len(transition_regex.findall(text)) / wc_safe
        contraction_density = len(contraction_regex.findall(text)) / wc_safe
        pronoun_density = len(pronoun_regex.findall(text)) / wc_safe
        question_density = text.count("?") / wc_safe
        exclamation_density = len(re.findall(r"!(?!=)", text)) / wc_safe
        ellipsis_density = len(re.findall(r"\.{2,}", text)) / wc_safe
        sent_len_var = calculate_sent_len_var(text, words)

        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        uppercase_word_ratio = upper_words / wc_safe

        irreg_caps = sum(1 for w in words if any(c.isupper() for c in w[1:]) and not w.isupper())
        irregular_caps_ratio = irreg_caps / wc_safe

        code_kw_count = len(code_kw_regex.findall(text))
        code_keyword_density = code_kw_count / wc_safe
        code_sym_count = len(code_sym_regex.findall(text))
        code_symbol_density = code_sym_count / wc_safe
        function_def_flag = 1.0 if bool(re.search(r"\bdef\s+[a-zA-Z_]\w*\s*\(", text)) else 0.0
        import_stmt_flag = 1.0 if bool(re.search(r"(\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import)", text)) else 0.0
        has_indentation = 1.0 if bool(re.search(r"^\s{2,}\S", text, re.MULTILINE)) else 0.0
        code_lines = sum(1 for l in text.splitlines() if re.search(r"(def\s+|class\s+|return|import|^\s{2,}\S|[{};])", l))
        code_line_ratio = code_lines / max(len(text.splitlines()), 1)
        code_likelihood = min(1.0, (code_keyword_density * 2.5 + code_symbol_density * 2.0 + function_def_flag * 0.4 + import_stmt_flag * 0.3 + has_indentation * 0.2 + code_line_ratio * 0.3))

        non_ascii_count = sum(1 for char in text if ord(char) > 127)
        non_ascii_ratio = non_ascii_count / cc_safe
        german_char_count = len(german_chars_regex.findall(text))
        german_char_density = german_char_count / cc_safe
        accented_char_count = len(accented_chars_regex.findall(text))
        accented_char_density = accented_char_count / cc_safe
        is_german_indicator = 1.0 if (german_char_count > 0 or bool(re.search(r"\b(der|die|das|und|ist|nicht|f[üu]r|ein|eine|einen|einer|vom|von|mit|sich|dem|den)\b", text, re.IGNORECASE))) else 0.0

        row = [
            log_word_count, sentence_count, avg_sentence_length, vocab_diversity, length_bucket,
            token_entropy, rep_ngram_ratio, transition_ratio, contraction_density, pronoun_density,
            question_density, exclamation_density, ellipsis_density, sent_len_var,
            uppercase_word_ratio, irregular_caps_ratio, code_keyword_density, code_symbol_density,
            function_def_flag, import_stmt_flag, code_line_ratio, code_likelihood,
            non_ascii_ratio, german_char_density, accented_char_density, is_german_indicator
        ]
        features.append(row)
    return np.asarray(features, dtype=np.float32)

print("\nExtracting Base TF-IDF & MiniLM Embeddings...")
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=150000, sublinear_tf=True, min_df=2, max_df=0.98)
char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=150000, sublinear_tf=True, min_df=2, max_df=0.98)
mchar_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000, sublinear_tf=True, min_df=2, max_df=0.98)

X_tr_w = word_vectorizer.fit_transform(model_train_df["text"])
X_cal_w = word_vectorizer.transform(calibration_df["text"])
X_val_w = word_vectorizer.transform(validation_df["text"])

X_tr_c = char_vectorizer.fit_transform(model_train_df["text"])
X_cal_c = char_vectorizer.transform(calibration_df["text"])
X_val_c = char_vectorizer.transform(validation_df["text"])

X_tr_mc = mchar_vectorizer.fit_transform(model_train_df["text"])
X_cal_mc = mchar_vectorizer.transform(calibration_df["text"])
X_val_mc = mchar_vectorizer.transform(validation_df["text"])

transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_tr_emb = transformer_model.encode(model_train_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_cal_emb = transformer_model.encode(calibration_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_val_emb = transformer_model.encode(validation_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

tr_wc = np.array([len(t.split()) for t in model_train_df["text"]], dtype=np.float32)
cal_wc = np.array([len(t.split()) for t in calibration_df["text"]], dtype=np.float32)
val_wc = np.array([len(t.split()) for t in validation_df["text"]], dtype=np.float32)

decay_tr = (0.75 * np.exp(-tr_wc / 15.0)).reshape(-1, 1).astype(np.float32)
decay_cal = (0.75 * np.exp(-cal_wc / 15.0)).reshape(-1, 1).astype(np.float32)
decay_val = (0.75 * np.exp(-val_wc / 15.0)).reshape(-1, 1).astype(np.float32)

X_tr_si = csr_matrix((X_tr_emb * decay_tr).astype(np.float32))
X_cal_si = csr_matrix((X_cal_emb * decay_cal).astype(np.float32))
X_val_si = csr_matrix((X_val_emb * decay_val).astype(np.float32))

scaler_base = StandardScaler()
X_tr_s26_raw = extract_base_scalar_features(model_train_df["text"].tolist())
X_cal_s26_raw = extract_base_scalar_features(calibration_df["text"].tolist())
X_val_s26_raw = extract_base_scalar_features(validation_df["text"].tolist())

X_tr_s26 = scaler_base.fit_transform(X_tr_s26_raw)
X_cal_s26 = scaler_base.transform(X_cal_s26_raw)
X_val_s26 = scaler_base.transform(X_val_s26_raw)

# Assemble Baseline Feature Space (194,897 dimensions)
X_train_base = hstack([X_tr_w, X_tr_c, X_tr_mc, csr_matrix(X_tr_emb), X_tr_si, csr_matrix(X_tr_s26)], format="csr")
X_cal_base = hstack([X_cal_w, X_cal_c, X_cal_mc, csr_matrix(X_cal_emb), X_cal_si, csr_matrix(X_cal_s26)], format="csr")
X_val_base = hstack([X_val_w, X_val_c, X_val_mc, csr_matrix(X_val_emb), X_val_si, csr_matrix(X_val_s26)], format="csr")

y_train = model_train_df["label"].values.astype(int)
y_calibration = calibration_df["label"].values.astype(int)
y_validation = validation_df["label"].values.astype(int)

length_weights = {"very_short": 1.25, "short": 1.15, "medium": 1.05, "long": 1.00}
train_sample_weights = model_train_df["length_group"].map(length_weights).astype(float).values

def compute_ece(probs, true_labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(true_labels[in_bin])
            avg_conf_in_bin = np.mean(probs[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return float(ece)

def optimize_length_thresholds(y_cal, p_cal, cal_groups):
    thresholds = {}
    for grp in ["very_short", "short", "medium", "long"]:
        mask = (cal_groups == grp)
        if not np.any(mask):
            thresholds[grp] = 0.50
            continue
        y_g = y_cal[mask]
        p_g = p_cal[mask]
        sweeps = []
        for t in np.arange(0.20, 0.81, 0.01):
            t_val = round(float(t), 2)
            preds = (p_g >= t_val).astype(int)
            f = f1_score(y_g, preds, zero_division=0)
            sweeps.append((f, t_val))
        max_f = max(s[0] for s in sweeps)
        plateau = [t for f, t in sweeps if f >= max_f - 0.005]
        thresholds[grp] = round(float(np.median(plateau)), 2)
    return thresholds

# ----------------------------------------------------------------------
# SYSTEMATIC CANDIDATE EXPERIMENTS
# ----------------------------------------------------------------------
v3_experiments = []

def evaluate_candidate(candidate_name, X_tr, X_cal, X_val, clf, sample_w=train_sample_weights, calib_type="platt"):
    print(f"\n=======================================================")
    print(f"Evaluating: {candidate_name}")
    print(f"=======================================================")
    if sample_w is not None:
        clf.fit(X_tr, y_train, sample_weight=sample_w)
    else:
        clf.fit(X_tr, y_train)
        
    if hasattr(clf, "decision_function"):
        cal_scores = clf.decision_function(X_cal).reshape(-1, 1)
        val_scores = clf.decision_function(X_val).reshape(-1, 1)
    else:
        cal_scores = clf.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        val_scores = clf.predict_proba(X_val)[:, 1].reshape(-1, 1)

    if calib_type == "platt":
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        calibrator.fit(cal_scores, y_calibration)
        p_cal = calibrator.predict_proba(cal_scores)[:, 1]
        p_val = calibrator.predict_proba(val_scores)[:, 1]
    elif calib_type == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_scores.ravel(), y_calibration)
        p_cal = calibrator.predict(cal_scores.ravel())
        p_val = calibrator.predict(val_scores.ravel())
    else:
        calibrator = None
        if hasattr(clf, "predict_proba"):
            p_cal = clf.predict_proba(X_cal)[:, 1]
            p_val = clf.predict_proba(X_val)[:, 1]
        else:
            p_cal = 1 / (1 + np.exp(-cal_scores.ravel()))
            p_val = 1 / (1 + np.exp(-val_scores.ravel()))

    cal_auc = roc_auc_score(y_calibration, p_cal)
    cal_brier = brier_score_loss(y_calibration, p_cal)
    cal_ece = compute_ece(p_cal, y_calibration)
    
    length_thresh = optimize_length_thresholds(y_calibration, p_cal, calibration_df["length_group"].values)
    
    val_applied_t = np.array([length_thresh[g] for g in validation_df["length_group"].values])
    val_preds = (p_val >= val_applied_t).astype(int)
    
    val_acc = accuracy_score(y_validation, val_preds)
    val_bal_acc = balanced_accuracy_score(y_validation, val_preds)
    val_prec = precision_score(y_validation, val_preds, zero_division=0)
    val_rec = recall_score(y_validation, val_preds, zero_division=0)
    val_f1 = f1_score(y_validation, val_preds, zero_division=0)
    val_auc = roc_auc_score(y_validation, p_val)
    val_brier = brier_score_loss(y_validation, p_val)
    val_ece = compute_ece(p_val, y_validation)
    
    cm = confusion_matrix(y_validation, val_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  Calib Split: AUC={cal_auc:.4f} | Brier={cal_brier:.5f} | ECE={cal_ece:.4f}")
    print(f"  Val Split  : Acc={val_acc*100:.2f}% | BalAcc={val_bal_acc*100:.2f}% | Prec={val_prec*100:.2f}% | Rec={val_rec*100:.2f}% | F1={val_f1*100:.2f}% | AUC={val_auc*100:.2f}%")
    print(f"  Confusion  : TN={tn}, FP={fp}, FN={fn}, TP={tp} | Total Errors={fp+fn}")
    print(f"  Thresholds : {length_thresh}")

    res = {
        "candidate": candidate_name,
        "cal_auc": round(float(cal_auc), 5),
        "cal_brier": round(float(cal_brier), 5),
        "cal_ece": round(float(cal_ece), 5),
        "val_acc": round(float(val_acc), 5),
        "val_bal_acc": round(float(val_bal_acc), 5),
        "val_prec": round(float(val_prec), 5),
        "val_rec": round(float(val_rec), 5),
        "val_f1": round(float(val_f1), 5),
        "val_auc": round(float(val_auc), 5),
        "val_brier": round(float(val_brier), 5),
        "val_ece": round(float(val_ece), 5),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "total_errors": int(fp + fn),
        "thresholds": length_thresh,
        "classifier": clf,
        "calibrator": calibrator,
        "calib_type": calib_type
    }
    v3_experiments.append(res)
    return res

# 1. Candidate A: Production Baseline (V3.4: C=3.0, liblinear, length_weights)
evaluate_candidate(
    "Candidate_A_V3.4_Baseline",
    X_train_base, X_cal_base, X_val_base,
    LogisticRegression(C=3.0, solver="liblinear", class_weight="balanced", random_state=42)
)

# 2. Candidate C: Comprehensive Regularization Sweep on C
for c in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
    evaluate_candidate(
        f"Candidate_C_LogReg_C_{c}",
        X_train_base, X_cal_base, X_val_base,
        LogisticRegression(C=c, solver="liblinear", class_weight="balanced", random_state=42)
    )

# 3. Candidate D: Linear SVM with Platt Calibration
for c_svm in [0.1, 0.5, 1.0, 2.0]:
    evaluate_candidate(
        f"Candidate_D_LinearSVM_C_{c_svm}",
        X_train_base, X_cal_base, X_val_base,
        LinearSVC(C=c_svm, class_weight="balanced", random_state=42, max_iter=4000)
    )

# 4. Candidate F: Logistic Regression with Feature Normalization (L2-normalized full matrix)
normalizer = Normalizer(norm="l2")
X_train_norm = normalizer.fit_transform(X_train_base)
X_cal_norm = normalizer.transform(X_cal_base)
X_val_norm = normalizer.transform(X_val_base)

for c_norm in [1.0, 2.0, 5.0, 10.0, 20.0]:
    evaluate_candidate(
        f"Candidate_F_L2Norm_LogReg_C_{c_norm}",
        X_train_norm, X_cal_norm, X_val_norm,
        LogisticRegression(C=c_norm, solver="liblinear", class_weight="balanced", random_state=42)
    )

# 5. Candidate G: Class Weighting Variants (Uniform vs Inverse Frequency)
evaluate_candidate(
    "Candidate_G_UniformWeights_C_3.0",
    X_train_base, X_cal_base, X_val_base,
    LogisticRegression(C=3.0, solver="liblinear", class_weight=None, random_state=42),
    sample_w=None
)

evaluate_candidate(
    "Candidate_G_BalancedClassOnly_C_3.0",
    X_train_base, X_cal_base, X_val_base,
    LogisticRegression(C=3.0, solver="liblinear", class_weight="balanced", random_state=42),
    sample_w=None
)

# ----------------------------------------------------------------------
# EXPERIMENT RESULTS SUMMARY
# ----------------------------------------------------------------------
summary_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ["classifier", "calibrator"]}
    for r in v3_experiments
])
summary_csv = os.path.join(REPORTS_DIR, "text_v3_experiments_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 80)
print("EXPERIMENTS SUMMARY TABLE (SORTED BY VAL F1 & ROC-AUC):")
print("=" * 80)
sorted_df = summary_df.sort_values(by=["val_f1", "val_auc", "val_acc"], ascending=False)
print(sorted_df[["candidate", "val_acc", "val_f1", "val_auc", "val_brier", "val_ece", "total_errors", "FP", "FN"]].to_string(index=False))

# ----------------------------------------------------------------------
# EXTERNAL GENERALIZATION BENCHMARKS
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXTERNAL BENCHMARK EVALUATION FOR TOP CANDIDATES")
print("=" * 80)

top_candidates = [
    ("V3.4_Baseline", [r for r in v3_experiments if r["candidate"] == "Candidate_A_V3.4_Baseline"][0], False),
    ("LogReg_C_4.0", [r for r in v3_experiments if r["candidate"] == "Candidate_C_LogReg_C_4.0"][0], False),
    ("LogReg_C_5.0", [r for r in v3_experiments if r["candidate"] == "Candidate_C_LogReg_C_5.0"][0], False)
]

ext_records = []
for cand_name, c_res, is_norm in top_candidates:
    clf = c_res["classifier"]
    calibrator = c_res["calibrator"]
    length_thresh = c_res["thresholds"]
    print(f"\nEvaluating: {cand_name}")
    print("-" * 50)
    
    for ext_name, ext_df in ext_dfs.items():
        texts = ext_df["text"].tolist()
        y_ext = ext_df["label"].values.astype(int)
        
        X_w = word_vectorizer.transform(texts)
        X_c = char_vectorizer.transform(texts)
        X_mc = mchar_vectorizer.transform(texts)
        emb = transformer_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        wcs = np.array([len(t.split()) for t in texts], dtype=np.float32)
        decays = (0.75 * np.exp(-wcs / 15.0)).reshape(-1, 1).astype(np.float32)
        X_si = csr_matrix((emb * decays).astype(np.float32))
        
        raw_s = extract_base_scalar_features(texts)
        X_s = scaler_base.transform(raw_s)
        
        X_full = hstack([X_w, X_c, X_mc, csr_matrix(emb), X_si, csr_matrix(X_s)], format="csr")
        if is_norm:
            X_full = normalizer.transform(X_full)
            
        scores = clf.decision_function(X_full).reshape(-1, 1)
        probs = calibrator.predict_proba(scores)[:, 1]
        
        lgs = ext_df["length_group"].values
        applied_t = np.array([length_thresh.get(g, 0.50) for g in lgs])
        preds = (probs >= applied_t).astype(int)
        
        ext_acc = accuracy_score(y_ext, preds)
        ext_prec = precision_score(y_ext, preds, zero_division=0)
        ext_rec = recall_score(y_ext, preds, zero_division=0)
        ext_f1 = f1_score(y_ext, preds, zero_division=0)
        try:
            ext_auc = roc_auc_score(y_ext, probs) if len(np.unique(y_ext)) > 1 else 0.0
        except Exception:
            ext_auc = 0.0
        ext_brier = brier_score_loss(y_ext, probs)
        cm = confusion_matrix(y_ext, preds)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        
        print(f"  [{ext_name:18s}] N={len(ext_df):5d} | Acc: {ext_acc*100:6.2f}% | F1: {ext_f1*100:6.2f}% | AUC: {ext_auc*100:6.2f}% | Brier: {ext_brier:.5f} | FP={fp}, FN={fn}")
        
        ext_records.append({
            "candidate": cand_name,
            "dataset": ext_name,
            "samples": len(ext_df),
            "accuracy": round(float(ext_acc), 4),
            "precision": round(float(ext_prec), 4),
            "recall": round(float(ext_rec), 4),
            "f1": round(float(ext_f1), 4),
            "roc_auc": round(float(ext_auc), 4),
            "brier": round(float(ext_brier), 5),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        })

ext_v3_df = pd.DataFrame(ext_records)
ext_v3_csv = os.path.join(REPORTS_DIR, "text_v3_external_generalization.csv")
ext_v3_df.to_csv(ext_v3_csv, index=False)
print(f"\nSaved External Generalization to {ext_v3_csv}")
print("=" * 80)
print("V3 EXPERIMENTAL SUITE COMPLETED!")
print("=" * 80)
