import os
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sentence_transformers import SentenceTransformer


# =========================================================
# V2 IMPROVED — SENTENCEAI THRESHOLD COMPARISON
# =========================================================

print()
print("=" * 60)
print("V2 IMPROVED EXTERNAL THRESHOLD COMPARISON")
print("=" * 60)
print()


# =========================================================
# SETTINGS
# =========================================================

MODEL_DIR = (
    r"models\text_final_v2_improved"
)

TEST_FILE = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_test.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


THRESHOLDS = [
    0.41,
    0.43,
    0.45,
    0.47,
    0.50,
    0.52,
    0.55
]


# =========================================================
# HELPERS
# =========================================================

def load_pickle(path):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(f)


def get_text_column(df):

    if "text" in df.columns:
        return "text"

    if "generation" in df.columns:
        return "generation"

    raise ValueError(
        "No text column found."
    )


def get_length_group(text):

    word_count = len(
        str(text).split()
    )

    if word_count <= 10:
        return "very_short"

    elif word_count <= 30:
        return "short"

    elif word_count <= 60:
        return "medium"

    else:
        return "long"


def build_linguistic_features(texts):

    features = []

    for text in texts:

        text = str(text)

        words = text.split()

        word_count = len(words)

        sentence_count = sum(
            1
            for char in text
            if char in ".!?"
        )

        if sentence_count == 0:
            sentence_count = 1

        avg_sentence_length = (
            word_count /
            max(sentence_count, 1)
        )

        if word_count > 0:

            unique_words = len(
                set(
                    word.lower()
                    for word in words
                )
            )

            vocabulary_diversity = (
                unique_words /
                word_count
            )

        else:

            vocabulary_diversity = 0.0

        log_word_count = np.log1p(
            word_count
        )

        if word_count <= 10:
            length_bucket = 0.0

        elif word_count <= 30:
            length_bucket = 1.0

        elif word_count <= 60:
            length_bucket = 2.0

        else:
            length_bucket = 3.0

        features.append(
            [
                log_word_count,
                sentence_count,
                avg_sentence_length,
                vocabulary_diversity,
                length_bucket
            ]
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# =========================================================
# LOAD DATA
# =========================================================

print()
print("LOADING SENTENCEAI EXTERNAL DATA")
print("================================")


df = pd.read_csv(
    TEST_FILE
)


text_column = get_text_column(
    df
)


df = df.copy()


df["text"] = (
    df[text_column]
    .fillna("")
    .astype(str)
    .str.strip()
)


df = df[
    df["text"].str.len() > 0
].copy()


# =========================================================
# LABEL NORMALIZATION
# =========================================================

if "label" not in df.columns:

    if "label_name" in df.columns:

        df["label"] = (
            df["label_name"]
            .map(
                {
                    "Human": 0,
                    "AI": 1
                }
            )
        )

    else:

        raise ValueError(
            "No label column found."
        )


else:

    if df["label"].dtype == object:

        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "human": 0,
                    "ai": 1,
                    "0": 0,
                    "1": 1
                }
            )
        )

    else:

        df["label"] = (
            df["label"]
            .astype(int)
        )


df = df[
    df["label"].isin(
        [0, 1]
    )
].copy()


df["length_group"] = (
    df["text"]
    .apply(get_length_group)
)


print(
    f"External samples: {len(df)}"
)


print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    df["label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
    .value_counts()
    .to_string()
)


print()
print("LENGTH DISTRIBUTION")
print("===================")


print(
    pd.crosstab(
        df["length_group"],
        df["label"]
        .map(
            {
                0: "Human",
                1: "AI"
            }
        )
    ).to_string()
)


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("LOADING V2 IMPROVED MODEL")
print("==========================")


word_vectorizer = load_pickle(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)


char_vectorizer = load_pickle(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)


linguistic_scaler = load_pickle(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)


classifier = load_pickle(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


config = load_pickle(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


expected_features = int(
    config["feature_count"]
)


print(
    "V2 Improved model loaded."
)


print(
    f"Expected feature count: "
    f"{expected_features}"
)


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = (
    word_vectorizer.transform(
        df["text"]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# =========================================================
# CHARACTER FEATURES
# =========================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        df["text"]
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# =========================================================
# MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")


embedder = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# =========================================================
# EMBEDDINGS
# =========================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


X_embedding = embedder.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    X_embedding.shape
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_linguistic_raw = (
    build_linguistic_features(
        df["text"].tolist()
    )
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic_raw
    )
)


print(
    "Linguistic shape:",
    X_linguistic.shape
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("COMBINING FEATURES")
print("==================")


X = hstack(
    [
        X_word,
        X_char,
        csr_matrix(X_embedding),
        csr_matrix(X_linguistic)
    ],
    format="csr"
)


print(
    "Combined feature shape:",
    X.shape
)


# =========================================================
# FEATURE VALIDATION
# =========================================================

print()
print("FEATURE VALIDATION")
print("==================")


if X.shape[1] != expected_features:

    raise RuntimeError(
        f"Expected {expected_features} "
        f"features but got {X.shape[1]}"
    )


if X.shape[1] != classifier.n_features_in_:

    raise RuntimeError(
        "Feature count does not match classifier."
    )


print(
    "Feature configuration verified."
)


# =========================================================
# PROBABILITIES
# =========================================================

print()
print("CREATING PREDICTION PROBABILITIES")
print("=================================")


y_true = (
    df["label"]
    .astype(int)
    .values
)


probabilities = (
    classifier.predict_proba(
        X
    )[:, 1]
)


# =========================================================
# THRESHOLD COMPARISON
# =========================================================

print()
print("=" * 60)
print("THRESHOLD RESULTS")
print("=" * 60)


print()

print(
    "Threshold   Accuracy    Precision   Recall      F1"
)

print(
    "-" * 65
)


all_results = []


for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        predictions
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        predictions
    )


    human_fp = int(
        cm[0, 1]
    )


    ai_fn = int(
        cm[1, 0]
    )


    all_results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "human_fp": human_fp,
            "ai_fn": ai_fn
        }
    )


    print(
        f"{threshold:.2f}        "
        f"{accuracy:.4f}      "
        f"{precision:.4f}      "
        f"{recall:.4f}      "
        f"{f1:.4f}"
    )


# =========================================================
# LENGTH ANALYSIS
# =========================================================

print()
print("=" * 60)
print("AI RECALL BY TEXT LENGTH")
print("=" * 60)


for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    print()
    print(
        f"THRESHOLD = {threshold:.2f}"
    )

    print(
        "-" * 40
    )


    for group in [
        "very_short",
        "short",
        "medium",
        "long"
    ]:

        mask = (
            df["length_group"]
            .values
            == group
        )


        ai_mask = (
            mask
            &
            (y_true == 1)
        )


        ai_samples = int(
            np.sum(ai_mask)
        )


        if ai_samples == 0:
            continue


        ai_recall = (
            np.sum(
                predictions[ai_mask] == 1
            )
            /
            ai_samples
        )


        print(
            f"{group:12s} "
            f"AI Recall={ai_recall:.4f} "
            f"AI Samples={ai_samples}"
        )


# =========================================================
# BEST THRESHOLDS
# =========================================================

results_df = pd.DataFrame(
    all_results
)


best_f1_row = (
    results_df
    .sort_values(
        [
            "f1",
            "accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)


best_accuracy_row = (
    results_df
    .sort_values(
        [
            "accuracy",
            "f1"
        ],
        ascending=False
    )
    .iloc[0]
)


# Balanced objective:
# We want good F1 while also avoiding
# excessive Human false positives.

results_df["balanced_objective"] = (
    results_df["f1"]
    -
    0.0002 *
    results_df["human_fp"]
)


best_balanced_row = (
    results_df
    .sort_values(
        [
            "balanced_objective",
            "accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)


print()
print("=" * 60)
print("BEST THRESHOLDS")
print("=" * 60)


print(
    f"Best F1 threshold       : "
    f"{best_f1_row['threshold']:.2f}"
)

print(
    f"Best F1                 : "
    f"{best_f1_row['f1']:.4f}"
)


print(
    f"Best Accuracy threshold : "
    f"{best_accuracy_row['threshold']:.2f}"
)

print(
    f"Best Accuracy           : "
    f"{best_accuracy_row['accuracy']:.4f}"
)


print(
    f"Balanced threshold      : "
    f"{best_balanced_row['threshold']:.2f}"
)

print(
    f"Balanced F1             : "
    f"{best_balanced_row['f1']:.4f}"
)

print(
    f"Human FP                : "
    f"{int(best_balanced_row['human_fp'])}"
)

print(
    f"AI FN                   : "
    f"{int(best_balanced_row['ai_fn'])}"
)


# =========================================================
# SAVE RESULTS
# =========================================================

OUTPUT_FILE = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_v2_improved_thresholds.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("RESULTS SAVED")
print("=============")


print(
    f"Output: {OUTPUT_FILE}"
)


print()
print("=" * 60)
print("V2 IMPROVED THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)