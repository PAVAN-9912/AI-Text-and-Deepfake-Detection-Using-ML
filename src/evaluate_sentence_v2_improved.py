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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sentence_transformers import SentenceTransformer


# =========================================================
# V2 IMPROVED — SHORT TEXT EXTERNAL EVALUATION
# =========================================================

print()
print("=" * 60)
print("V2 IMPROVED MODEL — SHORT-TEXT EXTERNAL TEST")
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


# =========================================================
# HELPER FUNCTIONS
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


def get_label_column(df):

    if "label" in df.columns:
        return "label"

    if "label_name" in df.columns:
        return "label_name"

    raise ValueError(
        "No label column found."
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
# LOAD TEST DATA
# =========================================================

print()
print("LOADING SHORT-TEXT TEST DATA")
print("============================")


test_df = pd.read_csv(
    TEST_FILE
)


text_column = get_text_column(
    test_df
)

label_column = get_label_column(
    test_df
)


test_df = test_df.copy()


test_df["text"] = (
    test_df[text_column]
    .fillna("")
    .astype(str)
    .str.strip()
)


test_df = test_df[
    test_df["text"].str.len() > 0
].copy()


# =========================================================
# NORMALIZE LABELS
# =========================================================

if label_column == "label_name":

    test_df["label"] = (
        test_df["label_name"]
        .map(
            {
                "Human": 0,
                "AI": 1
            }
        )
    )

else:

    if test_df["label"].dtype == object:

        test_df["label"] = (
            test_df["label"]
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

        test_df["label"] = (
            test_df["label"]
            .astype(int)
        )


test_df = test_df[
    test_df["label"].isin(
        [0, 1]
    )
].copy()


test_df["length_group"] = (
    test_df["text"]
    .apply(get_length_group)
)


print(
    f"External test samples: "
    f"{len(test_df)}"
)


print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    test_df["label"]
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
        test_df["length_group"],
        test_df["label"]
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


decision_threshold = float(
    config["decision_threshold"]
)


print(
    "V2 Improved model components loaded."
)


print(
    f"Expected feature count: "
    f"{expected_features}"
)


# =========================================================
# WORD TF-IDF
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = (
    word_vectorizer.transform(
        test_df["text"]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        test_df["text"]
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


print(
    "Transformer:",
    TRANSFORMER_NAME
)


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
    test_df["text"].tolist(),
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
        test_df["text"].tolist()
    )
)


print(
    "Raw linguistic shape:",
    X_linguistic_raw.shape
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic_raw
    )
)


print(
    "Scaled linguistic shape:",
    X_linguistic.shape
)


# =========================================================
# CONVERT TO SPARSE
# =========================================================

X_embedding_sparse = csr_matrix(
    X_embedding
)


X_linguistic_sparse = csr_matrix(
    X_linguistic
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
        X_embedding_sparse,
        X_linguistic_sparse
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


actual_features = X.shape[1]

classifier_features = (
    classifier.n_features_in_
)


print(
    f"Expected features: "
    f"{expected_features}"
)

print(
    f"Classifier features: "
    f"{classifier_features}"
)

print(
    f"Actual features: "
    f"{actual_features}"
)


if (
    actual_features !=
    expected_features
):

    raise RuntimeError(
        "Feature count does not match "
        "saved model configuration."
    )


if (
    actual_features !=
    classifier_features
):

    raise RuntimeError(
        "Feature count does not match "
        "classifier."
    )


print(
    "Feature configuration verified."
)


# =========================================================
# THRESHOLD
# =========================================================

print()
print(
    f"V2 Improved decision threshold: "
    f"{decision_threshold:.2f}"
)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("RUNNING V2 IMPROVED PREDICTIONS")
print("===============================")


probabilities = (
    classifier.predict_proba(
        X
    )[:, 1]
)


predictions = (
    probabilities >=
    decision_threshold
).astype(int)


y_true = (
    test_df["label"]
    .astype(int)
    .values
)


# =========================================================
# OVERALL METRICS
# =========================================================

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


roc_auc = roc_auc_score(
    y_true,
    probabilities
)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("V2 IMPROVED SHORT-TEXT EXTERNAL RESULTS")
print("=" * 60)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print("================")


print(
    confusion_matrix(
        y_true,
        predictions
    )
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("CLASSIFICATION REPORT")
print("=====================")


print(
    classification_report(
        y_true,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# =========================================================
# PERFORMANCE BY LENGTH
# =========================================================

print()
print("PERFORMANCE BY TEXT LENGTH")
print("===========================")


result_df = test_df.copy()


result_df["prediction"] = (
    predictions
)


result_df["probability"] = (
    probabilities
)


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = result_df[
        result_df["length_group"]
        == group
    ]


    if len(subset) == 0:
        continue


    group_accuracy = (
        accuracy_score(
            subset["label"],
            subset["prediction"]
        )
    )


    group_precision = (
        precision_score(
            subset["label"],
            subset["prediction"],
            zero_division=0
        )
    )


    group_recall = (
        recall_score(
            subset["label"],
            subset["prediction"],
            zero_division=0
        )
    )


    group_f1 = (
        f1_score(
            subset["label"],
            subset["prediction"],
            zero_division=0
        )
    )


    print(
        f"{group:12s} "
        f"Accuracy={group_accuracy:.4f} "
        f"Precision={group_precision:.4f} "
        f"Recall={group_recall:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={len(subset)}"
    )


# =========================================================
# AI RECALL BY LENGTH
# =========================================================

print()
print("AI DETECTION BY TEXT LENGTH")
print("===========================")


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = result_df[
        result_df["length_group"]
        == group
    ]


    ai_subset = subset[
        subset["label"] == 1
    ]


    if len(ai_subset) == 0:
        continue


    ai_recall = recall_score(
        ai_subset["label"],
        ai_subset["prediction"],
        zero_division=0
    )


    print(
        f"{group:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={len(ai_subset)}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

OUTPUT_FILE = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_v2_improved_final_results.csv"
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("RESULTS SAVED")
print("=============")


print(
    f"Output: {OUTPUT_FILE}"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

correct = int(
    np.sum(
        predictions == y_true
    )
)


incorrect = (
    len(y_true) -
    correct
)


print()
print("=" * 60)
print("V2 IMPROVED SHORT-TEXT FINAL SUMMARY")
print("=" * 60)


print(
    f"Total samples        : "
    f"{len(y_true)}"
)

print(
    f"Correct predictions  : "
    f"{correct}"
)

print(
    f"Incorrect predictions: "
    f"{incorrect}"
)

print(
    f"Accuracy             : "
    f"{accuracy:.4f}"
)

print(
    f"F1 Score             : "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC              : "
    f"{roc_auc:.4f}"
)


print()
print(
    "V2 IMPROVED SHORT-TEXT "
    "EXTERNAL EVALUATION COMPLETED"
)