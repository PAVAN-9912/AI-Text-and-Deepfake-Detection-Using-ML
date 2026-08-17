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


print()
print("FINAL MODEL — SHORT-TEXT EXTERNAL TEST")
print("=======================================")
print()


# =========================================================
# SETTINGS
# =========================================================

MODEL_DIR = r"models\text_final"

TEST_PATH = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_test.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def build_linguistic_features(texts):

    features = []

    for text in texts:

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


def load_pickle(path):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(f)


def get_length_group(word_count):

    if word_count <= 10:
        return "very_short"

    elif word_count <= 30:
        return "short"

    elif word_count <= 60:
        return "medium"

    else:
        return "long"


# =========================================================
# LOAD DATA
# =========================================================

print("LOADING SHORT-TEXT TEST DATA")
print("============================")


df = pd.read_csv(
    TEST_PATH
)


df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df["length_group"] = (
    df["text"]
    .apply(
        lambda x:
        get_length_group(
            len(x.split())
        )
    )
)


print(
    f"External test samples: "
    f"{len(df)}"
)


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label_name"]
    .value_counts()
    .to_string()
)


print()
print("LENGTH DISTRIBUTION")
print("===================")

print(
    pd.crosstab(
        df["length_group"],
        df["label_name"]
    )
    .to_string()
)


# =========================================================
# LOAD VECTORIZERS
# =========================================================

print()
print("LOADING TF-IDF MODELS")
print("=====================")


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


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = word_vectorizer.transform(
    df["text"]
)


print(
    f"Word feature shape: "
    f"{X_word.shape}"
)


# =========================================================
# CHARACTER FEATURES
# =========================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = char_vectorizer.transform(
    df["text"]
)


print(
    f"Character feature shape: "
    f"{X_char.shape}"
)


# =========================================================
# MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")


print(
    f"Transformer: "
    f"{TRANSFORMER_NAME}"
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
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    f"Embedding shape: "
    f"{X_embedding.shape}"
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


print(
    f"Raw linguistic shape: "
    f"{X_linguistic_raw.shape}"
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic_raw
    )
)


print(
    f"Scaled linguistic shape: "
    f"{X_linguistic.shape}"
)


# =========================================================
# COMBINE
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
    f"Combined feature shape: "
    f"{X.shape}"
)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


y_true = (
    df["label"]
    .astype(int)
    .values
)


y_pred = classifier.predict(
    X
)


y_probability = (
    classifier.predict_proba(
        X
    )[:, 1]
)


# =========================================================
# OVERALL METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


print()
print("SHORT-TEXT FINAL RESULTS")
print("========================")

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
        y_pred
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
        y_pred,
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


results = df.copy()

results["prediction"] = y_pred

results["ai_probability"] = y_probability


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = results[
        results["length_group"] == group
    ]


    if len(subset) == 0:
        continue


    group_accuracy = accuracy_score(
        subset["label"],
        subset["prediction"]
    )


    group_precision = precision_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    group_recall = recall_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    group_f1 = f1_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
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

    subset = results[
        (
            results["length_group"] == group
        )
        &
        (
            results["label"] == 1
        )
    ]


    if len(subset) == 0:
        continue


    ai_recall = (
        subset["prediction"]
        .sum()
        /
        len(subset)
    )


    print(
        f"{group:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={len(subset)}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

OUTPUT_PATH = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_final_results.csv"
)


results.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("RESULTS SAVED")
print("=============")

print(
    f"Output: {OUTPUT_PATH}"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

correct = int(
    (y_true == y_pred).sum()
)

incorrect = int(
    (y_true != y_pred).sum()
)


print()
print("FINAL SHORT-TEXT SUMMARY")
print("========================")

print(
    f"Total samples        : {len(df)}"
)

print(
    f"Correct predictions  : {correct}"
)

print(
    f"Incorrect predictions: {incorrect}"
)

print(
    f"Accuracy             : {accuracy:.4f}"
)

print(
    f"F1 Score             : {f1:.4f}"
)

print(
    f"ROC-AUC              : {roc_auc:.4f}"
)


print()
print(
    "SHORT-TEXT FINAL EVALUATION COMPLETED"
)