import os
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sentence_transformers import SentenceTransformer


# ============================================================
# V2 IMPROVED — HC3 THRESHOLD COMPARISON
# ============================================================

print()
print("=" * 60)
print("V2 IMPROVED HC3 THRESHOLD COMPARISON")
print("=" * 60)


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = r"models\text_final_v2_improved"

HC3_VALIDATION = r"datasets\text\HC3\validation.csv"

OUTPUT_FILE = (
    r"datasets\text\HC3"
    r"\hc3_v2_improved_thresholds.csv"
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)


def get_text_column(df):

    if "text" in df.columns:
        return "text"

    if "generation" in df.columns:
        return "generation"

    raise ValueError(
        "No text column found."
    )


def clean_dataframe(df):

    df = df.copy()

    text_column = get_text_column(df)

    df["text"] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"].str.len() > 0
    ].copy()

    return df


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


def add_length_group(df):

    df = df.copy()

    df["length_group"] = (
        df["text"]
        .apply(get_length_group)
    )

    return df


def build_linguistic_features(texts):

    features = []

    for text in texts:

        text = str(text)

        words = text.split()

        word_count = len(words)

        # Sentence count
        sentence_count = sum(
            1
            for char in text
            if char in ".!?"
        )

        if sentence_count == 0:
            sentence_count = 1

        # Average sentence length
        avg_sentence_length = (
            word_count /
            max(sentence_count, 1)
        )

        # Vocabulary diversity
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

        # Log word count
        log_word_count = np.log1p(
            word_count
        )

        # Length bucket
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


# ============================================================
# LOAD HC3
# ============================================================

print()
print("LOADING HC3 VALIDATION DATA")
print("===========================")


hc3 = pd.read_csv(
    HC3_VALIDATION
)


hc3 = clean_dataframe(
    hc3
)


hc3 = add_length_group(
    hc3
)


# ============================================================
# NORMALIZE LABELS
# ============================================================

if hc3["label"].dtype == object:

    hc3["label"] = (
        hc3["label"]
        .map(
            {
                "Human": 0,
                "AI": 1,
                "human": 0,
                "ai": 1
            }
        )
    )


hc3 = hc3[
    hc3["label"].isin(
        [0, 1]
    )
].copy()


hc3["label"] = (
    hc3["label"]
    .astype(int)
)


print(
    "Samples:",
    len(hc3)
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    hc3["label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
    .value_counts()
    .to_string()
)


# ============================================================
# LENGTH DISTRIBUTION
# ============================================================

print()
print("TEXT LENGTH DISTRIBUTION")
print("========================")


print(
    pd.crosstab(
        hc3["length_group"],
        hc3["label"]
    )
    .rename(
        columns={
            0: "Human",
            1: "AI"
        }
    )
    .to_string()
)


# ============================================================
# LOAD MODEL
# ============================================================

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


print(
    "V2 Improved model loaded."
)


expected_features = config.get(
    "feature_count"
)


print(
    "Expected feature count:",
    expected_features
)


# ============================================================
# WORD FEATURES
# ============================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = (
    word_vectorizer.transform(
        hc3["text"]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# ============================================================
# CHARACTER FEATURES
# ============================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        hc3["text"]
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# ============================================================
# MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


transformer_name = config.get(
    "transformer",
    "sentence-transformers/all-MiniLM-L6-v2"
)


print(
    "Transformer:",
    transformer_name
)


embedder = SentenceTransformer(
    transformer_name
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# EMBEDDINGS
# ============================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


X_embedding = embedder.encode(
    hc3["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    X_embedding.shape
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_linguistic_raw = (
    build_linguistic_features(
        hc3["text"].tolist()
    )
)


print(
    "Linguistic shape:",
    X_linguistic_raw.shape
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic_raw
    )
)


# ============================================================
# COMBINE FEATURES
# ============================================================

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


# ============================================================
# FEATURE VALIDATION
# ============================================================

print()
print("FEATURE VALIDATION")
print("==================")


actual_features = X.shape[1]

classifier_features = (
    classifier.n_features_in_
)


print(
    "Expected features:",
    expected_features
)

print(
    "Classifier features:",
    classifier_features
)

print(
    "Actual features:",
    actual_features
)


if (
    expected_features != classifier_features
    or
    actual_features != classifier_features
):

    raise RuntimeError(
        "Feature configuration mismatch."
    )


print(
    "Feature configuration verified."
)


# ============================================================
# CREATE PROBABILITIES
# ============================================================

print()
print("CREATING PREDICTION PROBABILITIES")
print("=================================")


y_true = (
    hc3["label"]
    .astype(int)
    .values
)


probabilities = (
    classifier.predict_proba(
        X
    )[:, 1]
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print()
print("=" * 60)
print("HC3 THRESHOLD RESULTS")
print("=" * 60)


print()
print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Human FP':<12}"
    f"{'AI FN':<12}"
)


print("-" * 90)


results = []


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


    human_fp = int(
        (
            (y_true == 0)
            &
            (predictions == 1)
        ).sum()
    )


    ai_fn = int(
        (
            (y_true == 1)
            &
            (predictions == 0)
        ).sum()
    )


    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{human_fp:<12d}"
        f"{ai_fn:<12d}"
    )


    results.append(
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


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST THRESHOLDS
# ============================================================

best_f1_row = (
    results_df
    .loc[
        results_df["f1"].idxmax()
    ]
)


best_accuracy_row = (
    results_df
    .loc[
        results_df["accuracy"].idxmax()
    ]
)


# Balanced objective:
#
# We want good F1 while also avoiding excessive
# false positives and false negatives.
#
# The objective uses the average of:
#   F1
#   Accuracy
#
# This prevents choosing a threshold only because
# it maximizes one metric.
# ============================================================

results_df["balanced_score"] = (
    (
        results_df["f1"]
        +
        results_df["accuracy"]
    )
    / 2.0
)


best_balanced_row = (
    results_df
    .loc[
        results_df["balanced_score"].idxmax()
    ]
)


print()
print("=" * 60)
print("BEST HC3 THRESHOLDS")
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
    f"Balanced score          : "
    f"{best_balanced_row['balanced_score']:.4f}"
)


print(
    f"Human FP               : "
    f"{int(best_balanced_row['human_fp'])}"
)


print(
    f"AI FN                  : "
    f"{int(best_balanced_row['ai_fn'])}"
)


# ============================================================
# AI RECALL BY LENGTH
# ============================================================

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
            (
                hc3["length_group"]
                == group
            )
            &
            (
                hc3["label"]
                == 1
            )
        )


        ai_samples = int(
            mask.sum()
        )


        if ai_samples == 0:
            continue


        ai_recall = (
            predictions[mask.values]
            == 1
        ).mean()


        print(
            f"{group:12s} "
            f"AI Recall={ai_recall:.4f} "
            f"AI Samples={ai_samples}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

print()
print("RESULTS SAVED")
print("=============")


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("V2 IMPROVED HC3 THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)