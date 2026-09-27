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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
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

EXPERIMENTS_DIR = os.path.join(BASE_DIR, "models", "text_experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("PHASE C & D & E & F — TEXT DETECTION EXPERIMENT PIPELINE")
print("=" * 80)

# ----------------------------------------------------------------------
# 1. LOAD DATASETS & RECONSTRUCT SPLITS
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

# ----------------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE
# ----------------------------------------------------------------------

# Regex definitions
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

STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
}

def calculate_token_entropy(words):
    if not words: return 0.0
    lengths = [len(w) for w in words]
    counts = Counter(lengths)
    total = len(lengths)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))

def calculate_char_entropy(text):
    if not text: return 0.0
    counts = Counter(text)
    total = len(text)
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

def extract_scalar_features(texts, enhanced=False):
    """
    Extracts base 26 features if enhanced=False.
    Extracts enhanced 34 features if enhanced=True (adds 8 robust short-text & stylometric descriptors).
    """
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

        if enhanced:
            # 8 Robust Short-Text & Stylometric Extensions
            char_entropy = calculate_char_entropy(text)
            wlens = [len(w) for w in words] if words else [0]
            avg_word_len = float(np.mean(wlens))
            std_word_len = float(np.std(wlens)) if len(wlens) > 1 else 0.0
            stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
            stopword_ratio = stopword_count / wc_safe
            digit_count = sum(1 for c in text if c.isdigit())
            digit_density = digit_count / cc_safe
            punct_count = sum(1 for c in text if c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
            punct_density = punct_count / cc_safe
            ttr_root = len(set(w.lower() for w in words)) / math.sqrt(wc_safe)
            hapax_count = sum(1 for _, cnt in Counter(w.lower() for w in words).items() if cnt == 1)
            hapax_ratio = hapax_count / wc_safe

            row.extend([
                char_entropy, avg_word_len, std_word_len, stopword_ratio,
                digit_density, punct_density, ttr_root, hapax_ratio
            ])

        features.append(row)
    return np.asarray(features, dtype=np.float32)

print("\nExtracting Base TF-IDF & Embeddings...")
# TF-IDF Vectorizers
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

# MiniLM
transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_tr_emb = transformer_model.encode(model_train_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_cal_emb = transformer_model.encode(calibration_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
X_val_emb = transformer_model.encode(validation_df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

# Decay Interactions
tr_wc = np.array([len(t.split()) for t in model_train_df["text"]], dtype=np.float32)
cal_wc = np.array([len(t.split()) for t in calibration_df["text"]], dtype=np.float32)
val_wc = np.array([len(t.split()) for t in validation_df["text"]], dtype=np.float32)

decay_tr = (0.75 * np.exp(-tr_wc / 15.0)).reshape(-1, 1).astype(np.float32)
decay_cal = (0.75 * np.exp(-cal_wc / 15.0)).reshape(-1, 1).astype(np.float32)
decay_val = (0.75 * np.exp(-val_wc / 15.0)).reshape(-1, 1).astype(np.float32)

X_tr_si = csr_matrix((X_tr_emb * decay_tr).astype(np.float32))
X_cal_si = csr_matrix((X_cal_emb * decay_cal).astype(np.float32))
X_val_si = csr_matrix((X_val_emb * decay_val).astype(np.float32))

# Scalar Base (26)
scaler_base = StandardScaler()
X_tr_s26_raw = extract_scalar_features(model_train_df["text"].tolist(), enhanced=False)
X_cal_s26_raw = extract_scalar_features(calibration_df["text"].tolist(), enhanced=False)
X_val_s26_raw = extract_scalar_features(validation_df["text"].tolist(), enhanced=False)

X_tr_s26 = scaler_base.fit_transform(X_tr_s26_raw)
X_cal_s26 = scaler_base.transform(X_cal_s26_raw)
X_val_s26 = scaler_base.transform(X_val_s26_raw)

# Scalar Enhanced (34)
scaler_enh = StandardScaler()
X_tr_s34_raw = extract_scalar_features(model_train_df["text"].tolist(), enhanced=True)
X_cal_s34_raw = extract_scalar_features(calibration_df["text"].tolist(), enhanced=True)
X_val_s34_raw = extract_scalar_features(validation_df["text"].tolist(), enhanced=True)

X_tr_s34 = scaler_enh.fit_transform(X_tr_s34_raw)
X_cal_s34 = scaler_enh.transform(X_cal_s34_raw)
X_val_s34 = scaler_enh.transform(X_val_s34_raw)

# Combined Baseline (26 features)
X_train_baseline = hstack([X_tr_w, X_tr_c, X_tr_mc, csr_matrix(X_tr_emb), X_tr_si, csr_matrix(X_tr_s26)], format="csr")
X_cal_baseline = hstack([X_cal_w, X_cal_c, X_cal_mc, csr_matrix(X_cal_emb), X_cal_si, csr_matrix(X_cal_s26)], format="csr")
X_val_baseline = hstack([X_val_w, X_val_c, X_val_mc, csr_matrix(X_val_emb), X_val_si, csr_matrix(X_val_s26)], format="csr")

# Combined Enhanced (34 features)
X_train_enhanced = hstack([X_tr_w, X_tr_c, X_tr_mc, csr_matrix(X_tr_emb), X_tr_si, csr_matrix(X_tr_s34)], format="csr")
X_cal_enhanced = hstack([X_cal_w, X_cal_c, X_cal_mc, csr_matrix(X_cal_emb), X_cal_si, csr_matrix(X_cal_s34)], format="csr")
X_val_enhanced = hstack([X_val_w, X_val_c, X_val_mc, csr_matrix(X_val_emb), X_val_si, csr_matrix(X_val_s34)], format="csr")

y_train = model_train_df["label"].values.astype(int)
y_calibration = calibration_df["label"].values.astype(int)
y_validation = validation_df["label"].values.astype(int)

length_weights = {"very_short": 1.25, "short": 1.15, "medium": 1.05, "long": 1.00}
train_sample_weights = model_train_df["length_group"].map(length_weights).astype(float).values

# External Test Sets Preprocessing
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

print(f"\nExternal Datasets Loaded: {list(ext_dfs.keys())}")

# Helper: compute ECE (Expected Calibration Error)
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
        best_f1, best_t = -1.0, 0.50
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

# ======================================================================
# EXPERIMENT SUITE
# ======================================================================
experiments_results = []

def run_experiment(exp_name, X_tr, X_cal, X_val, y_tr, y_cal, y_val, sample_w, clf, use_calibrator=True, calib_type="platt"):
    print(f"\n--- Running: {exp_name} ---")
    if sample_w is not None:
        clf.fit(X_tr, y_tr, sample_weight=sample_w)
    else:
        clf.fit(X_tr, y_tr)
    
    # Get raw decision scores / probabilities on calibration split
    if hasattr(clf, "decision_function"):
        cal_scores = clf.decision_function(X_cal).reshape(-1, 1)
        val_scores = clf.decision_function(X_val).reshape(-1, 1)
    else:
        cal_scores = clf.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        val_scores = clf.predict_proba(X_val)[:, 1].reshape(-1, 1)
        
    if use_calibrator and calib_type == "platt":
        # Fit Platt scaler strictly on calibration split
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        calibrator.fit(cal_scores, y_cal)
        p_cal = calibrator.predict_proba(cal_scores)[:, 1]
        p_val = calibrator.predict_proba(val_scores)[:, 1]
    elif use_calibrator and calib_type == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_scores.ravel(), y_cal)
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

    # Calibration split evaluation
    cal_auc = roc_auc_score(y_cal, p_cal)
    cal_brier = brier_score_loss(y_cal, p_cal)
    cal_ece = compute_ece(p_cal, y_cal)
    
    # Thresholds tuned strictly on calibration split
    length_thresh = optimize_length_thresholds(y_cal, p_cal, calibration_df["length_group"].values)
    
    # Predict on Calibration split
    cal_applied_t = np.array([length_thresh[g] for g in calibration_df["length_group"].values])
    cal_preds = (p_cal >= cal_applied_t).astype(int)
    cal_acc = accuracy_score(y_cal, cal_preds)
    cal_f1 = f1_score(y_cal, cal_preds, zero_division=0)
    
    # Validation evaluation (held-out)
    val_auc = roc_auc_score(y_val, p_val)
    val_brier = brier_score_loss(y_val, p_val)
    val_ece = compute_ece(p_val, y_val)
    val_applied_t = np.array([length_thresh[g] for g in validation_df["length_group"].values])
    val_preds = (p_val >= val_applied_t).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds, zero_division=0)
    val_rec = recall_score(y_val, val_preds, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    cm = confusion_matrix(y_val, val_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Calib AUC: {cal_auc:.4f} | Calib F1: {cal_f1:.4f} | Calib Brier: {cal_brier:.5f} | Calib ECE: {cal_ece:.4f}")
    print(f"  Val   Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Val AUC: {val_auc*100:.2f}% | FP={fp}, FN={fn} | Brier: {val_brier:.5f}")
    print(f"  Thresholds: {length_thresh}")

    res = {
        "experiment": exp_name,
        "cal_auc": round(float(cal_auc), 5),
        "cal_f1": round(float(cal_f1), 5),
        "cal_acc": round(float(cal_acc), 5),
        "cal_brier": round(float(cal_brier), 5),
        "cal_ece": round(float(cal_ece), 5),
        "val_acc": round(float(val_acc), 5),
        "val_prec": round(float(val_prec), 5),
        "val_rec": round(float(val_rec), 5),
        "val_f1": round(float(val_f1), 5),
        "val_auc": round(float(val_auc), 5),
        "val_brier": round(float(val_brier), 5),
        "val_ece": round(float(val_ece), 5),
        "FP": int(fp),
        "FN": int(fn),
        "thresholds": length_thresh,
        "classifier": clf,
        "calibrator": calibrator,
        "calib_type": calib_type
    }
    experiments_results.append(res)
    return res

# ----------------------------------------------------------------------
# EXPERIMENT 1: V3.4 Baseline
# ----------------------------------------------------------------------
run_experiment(
    "Exp1_V3.4_Baseline",
    X_train_baseline, X_cal_baseline, X_val_baseline,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=3.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="platt"
)

# ----------------------------------------------------------------------
# EXPERIMENT 2: Regularization Search (C in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
# ----------------------------------------------------------------------
c_values = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
for c_val in c_values:
    run_experiment(
        f"Exp2_RegSearch_C_{c_val}",
        X_train_baseline, X_cal_baseline, X_val_baseline,
        y_train, y_calibration, y_validation,
        sample_w=train_sample_weights,
        clf=LogisticRegression(C=c_val, solver="liblinear", class_weight="balanced", random_state=42),
        use_calibrator=True, calib_type="platt"
    )

# ----------------------------------------------------------------------
# EXPERIMENT 3: Solver Comparison (liblinear vs lbfgs vs saga)
# ----------------------------------------------------------------------
for solver in ["lbfgs", "liblinear"]:
    run_experiment(
        f"Exp3_Solver_{solver}_C_2.0",
        X_train_baseline, X_cal_baseline, X_val_baseline,
        y_train, y_calibration, y_validation,
        sample_w=train_sample_weights,
        clf=LogisticRegression(C=2.0, solver=solver, class_weight="balanced", max_iter=2500, random_state=42),
        use_calibrator=True, calib_type="platt"
    )

# ----------------------------------------------------------------------
# EXPERIMENT 4: Linear SVM with Platt Calibration
# ----------------------------------------------------------------------
run_experiment(
    "Exp4_LinearSVM_Platt",
    X_train_baseline, X_cal_baseline, X_val_baseline,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000),
    use_calibrator=True, calib_type="platt"
)

# ----------------------------------------------------------------------
# EXPERIMENT 5: Feature Ablations
# ----------------------------------------------------------------------
# A: Word TF-IDF only
run_experiment(
    "Exp5A_Word_TFIDF_Only",
    X_tr_w, X_cal_w, X_val_w,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="platt"
)

# B: Char TF-IDF only
run_experiment(
    "Exp5B_Char_TFIDF_Only",
    X_tr_c, X_cal_c, X_val_c,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="platt"
)

# C: Word + Char TF-IDF
X_tr_wc = hstack([X_tr_w, X_tr_c], format="csr")
X_cal_wc = hstack([X_cal_w, X_cal_c], format="csr")
X_val_wc = hstack([X_val_w, X_val_c], format="csr")
run_experiment(
    "Exp5C_Word_Char_TFIDF",
    X_tr_wc, X_cal_wc, X_val_wc,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="platt"
)

# D: TF-IDF + MiniLM
X_tr_wcm = hstack([X_tr_w, X_tr_c, X_tr_mc, csr_matrix(X_tr_emb)], format="csr")
X_cal_wcm = hstack([X_cal_w, X_cal_c, X_cal_mc, csr_matrix(X_cal_emb)], format="csr")
X_val_wcm = hstack([X_val_w, X_val_c, X_val_mc, csr_matrix(X_val_emb)], format="csr")
run_experiment(
    "Exp5D_TFIDF_MiniLM",
    X_tr_wcm, X_cal_wcm, X_val_wcm,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="platt"
)

# ----------------------------------------------------------------------
# EXPERIMENT 6: Improved Short-Text Representation (Enhanced 34 features)
# ----------------------------------------------------------------------
for c_enh in [1.5, 2.0, 2.5, 3.0]:
    run_experiment(
        f"Exp6_Enhanced_Stylometry_C_{c_enh}",
        X_train_enhanced, X_cal_enhanced, X_val_enhanced,
        y_train, y_calibration, y_validation,
        sample_w=train_sample_weights,
        clf=LogisticRegression(C=c_enh, solver="liblinear", class_weight="balanced", random_state=42),
        use_calibrator=True, calib_type="platt"
    )

# ----------------------------------------------------------------------
# EXPERIMENT 7: Class Weighting & Sample Weighting Variants
# ----------------------------------------------------------------------
# Uniform weights
run_experiment(
    "Exp7_Enhanced_UniformWeights_C_2.0",
    X_train_enhanced, X_cal_enhanced, X_val_enhanced,
    y_train, y_calibration, y_validation,
    sample_w=None,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight=None, random_state=42),
    use_calibrator=True, calib_type="platt"
)

# Balanced class weight only
run_experiment(
    "Exp7_Enhanced_BalancedClassOnly_C_2.0",
    X_train_enhanced, X_cal_enhanced, X_val_enhanced,
    y_train, y_calibration, y_validation,
    sample_w=None,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="platt"
)

# ----------------------------------------------------------------------
# PHASE D: Calibration Comparison (Raw vs Platt vs Isotonic)
# ----------------------------------------------------------------------
run_experiment(
    "Exp_Calib_Raw_Probs",
    X_train_enhanced, X_cal_enhanced, X_val_enhanced,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=False, calib_type="none"
)

run_experiment(
    "Exp_Calib_Isotonic",
    X_train_enhanced, X_cal_enhanced, X_val_enhanced,
    y_train, y_calibration, y_validation,
    sample_w=train_sample_weights,
    clf=LogisticRegression(C=2.0, solver="liblinear", class_weight="balanced", random_state=42),
    use_calibrator=True, calib_type="isotonic"
)

# ----------------------------------------------------------------------
# SAVE EXPERIMENT SUMMARY TABLE
# ----------------------------------------------------------------------
exp_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ["classifier", "calibrator"]}
    for r in experiments_results
])
exp_summary_csv = os.path.join(REPORTS_DIR, "text_experiments_summary.csv")
exp_df.to_csv(exp_summary_csv, index=False)
print(f"\nSaved Experiment Summary to {exp_summary_csv}")
print("\n" + "=" * 80)
print("EXPERIMENTS SUMMARY TABLE:")
print("=" * 80)
print(exp_df[["experiment", "cal_auc", "cal_f1", "val_acc", "val_f1", "val_auc", "val_brier", "FP", "FN"]].to_string(index=False))

# ----------------------------------------------------------------------
# PHASE F: EVALUATE BEST CANDIDATE ON EXTERNAL BENCHMARKS
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE F — EXTERNAL BENCHMARK EVALUATION FOR TOP MODELS")
print("=" * 80)

# Select top 2 models: Baseline (Exp1) and Best Enhanced Model (Exp6_C_2.0)
models_to_eval_external = [
    ("V3.4_Baseline", [r for r in experiments_results if r["experiment"] == "Exp1_V3.4_Baseline"][0], False),
    ("Enhanced_V2_Candidate", [r for r in experiments_results if r["experiment"] == "Exp6_Enhanced_Stylometry_C_2.0"][0], True)
]

external_eval_records = []

for model_name, m_res, is_enhanced in models_to_eval_external:
    clf = m_res["classifier"]
    calibrator = m_res["calibrator"]
    length_thresh = m_res["thresholds"]
    scaler = scaler_enh if is_enhanced else scaler_base
    
    print(f"\nEvaluating Model: {model_name} (Enhanced={is_enhanced})")
    print("-" * 60)
    
    for ext_name, ext_df in ext_dfs.items():
        texts = ext_df["text"].tolist()
        y_ext = ext_df["label"].values.astype(int)
        
        # Extract features
        X_w = word_vectorizer.transform(texts)
        X_c = char_vectorizer.transform(texts)
        X_mc = mchar_vectorizer.transform(texts)
        emb = transformer_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        wcs = np.array([len(t.split()) for t in texts], dtype=np.float32)
        decays = (0.75 * np.exp(-wcs / 15.0)).reshape(-1, 1).astype(np.float32)
        X_si = csr_matrix((emb * decays).astype(np.float32))
        
        raw_s = extract_scalar_features(texts, enhanced=is_enhanced)
        X_s = scaler.transform(raw_s)
        
        X_full = hstack([X_w, X_c, X_mc, csr_matrix(emb), X_si, csr_matrix(X_s)], format="csr")
        
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
        
        external_eval_records.append({
            "model": model_name,
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

ext_eval_df = pd.DataFrame(external_eval_records)
ext_eval_csv = os.path.join(REPORTS_DIR, "text_external_generalization.csv")
ext_eval_df.to_csv(ext_eval_csv, index=False)
print(f"\nSaved External Generalization to {ext_eval_csv}")

# ----------------------------------------------------------------------
# SAVE TOP ENHANCED ARTIFACTS FOR MODEL SELECTION (PHASE H)
# ----------------------------------------------------------------------
best_enhanced_res = [r for r in experiments_results if r["experiment"] == "Exp6_Enhanced_Stylometry_C_2.0"][0]
best_clf = best_enhanced_res["classifier"]
best_calibrator = best_enhanced_res["calibrator"]
best_thresholds = best_enhanced_res["thresholds"]

NEW_MODEL_DIR = os.path.join(BASE_DIR, "models", "text_final_v2")
os.makedirs(NEW_MODEL_DIR, exist_ok=True)

with open(os.path.join(NEW_MODEL_DIR, "word_tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(word_vectorizer, f)
with open(os.path.join(NEW_MODEL_DIR, "char_tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(char_vectorizer, f)
with open(os.path.join(NEW_MODEL_DIR, "multi_char_tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(mchar_vectorizer, f)
with open(os.path.join(NEW_MODEL_DIR, "linguistic_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_enh, f)
with open(os.path.join(NEW_MODEL_DIR, "text_classifier.pkl"), "wb") as f:
    pickle.dump(best_clf, f)
with open(os.path.join(NEW_MODEL_DIR, "probability_calibrator.pkl"), "wb") as f:
    pickle.dump({"model": best_calibrator, "method": "platt_sigmoid"}, f)

FEATURE_NAMES_V2 = [
    # 5 Base Linguistic
    "log_word_count", "sentence_count", "avg_sentence_length", "vocabulary_diversity", "length_bucket",
    # 11 Style / Lexical
    "token_entropy", "rep_ngram_ratio", "transition_ratio", "contraction_density", "pronoun_density",
    "question_density", "exclamation_density", "ellipsis_density", "sent_len_var", "uppercase_word_ratio", "irregular_caps_ratio",
    # 6 Code / Syntax
    "code_keyword_density", "code_symbol_density", "function_def_flag", "import_stmt_flag", "code_line_ratio", "code_likelihood",
    # 4 Multilingual
    "non_ascii_ratio", "german_char_density", "accented_char_density", "is_german_indicator",
    # 8 Enhanced Short-Text & Stylometry
    "char_entropy", "avg_word_length", "std_word_length", "stopword_ratio",
    "digit_density", "punct_density", "ttr_root", "hapax_ratio"
]

config_v2 = {
    "model_version": "TextDetector_V2",
    "transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "word_max_features": 150000,
    "char_max_features": 150000,
    "multi_char_max_features": 5000,
    "linguistic_feature_names": FEATURE_NAMES_V2[:5],
    "style_feature_names": FEATURE_NAMES_V2[5:16],
    "code_feature_names": FEATURE_NAMES_V2[16:22],
    "multilingual_feature_names": FEATURE_NAMES_V2[22:26],
    "enhanced_short_features": FEATURE_NAMES_V2[26:],
    "all_feature_names": FEATURE_NAMES_V2,
    "total_scalar_features": len(FEATURE_NAMES_V2),
    "short_interaction_alpha": 0.75,
    "classifier_C": 2.0,
    "random_seed": RANDOM_SEED,
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df),
    "training_datasets": ["HC3", "RAID", "SentenceAI"],
    "decision_threshold": best_thresholds,
    "length_thresholds": best_thresholds,
    "calibration_method": "platt_sigmoid"
}
with open(os.path.join(NEW_MODEL_DIR, "text_config.pkl"), "wb") as f:
    pickle.dump(config_v2, f)

metrics_v2 = {
    "model_version": "TextDetector_V2",
    "accuracy": best_enhanced_res["val_acc"],
    "precision": best_enhanced_res["val_prec"],
    "recall": best_enhanced_res["val_rec"],
    "f1": best_enhanced_res["val_f1"],
    "roc_auc": best_enhanced_res["val_auc"],
    "brier_score": best_enhanced_res["val_brier"],
    "ece": best_enhanced_res["val_ece"],
    "decision_threshold": best_thresholds,
    "length_thresholds": best_thresholds,
    "false_positives": best_enhanced_res["FP"],
    "false_negatives": best_enhanced_res["FN"],
    "training_samples": len(model_train_df),
    "calibration_samples": len(calibration_df),
    "validation_samples": len(validation_df)
}
with open(os.path.join(NEW_MODEL_DIR, "text_metrics.pkl"), "wb") as f:
    pickle.dump(metrics_v2, f)

print(f"\nSaved TextDetector_V2 Artifacts to: {NEW_MODEL_DIR}")
print("=" * 80)
print("EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
