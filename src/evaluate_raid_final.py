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
print("FINAL MODEL — RAID EXTERNAL TEST")
print("================================")
print()


# =========================================================
# SETTINGS
# =========================================================

MODEL_DIR = r"models\text_final"

RAID_PATH = (
    r"datasets\text\RAID"
    r"\raid_external.csv"
)

OUTPUT_PATH = (
    r"datasets\text\RAID"
    r"\raid_final_results.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)


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

        # Same length feature used during
        # final model training.
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
# LOAD RAID DATA
# =========================================================

print("LOADING RAID EXTERNAL DATA")
print("===========================")


df = pd.read_csv(
    RAID_PATH
)


# ---------------------------------------------------------
# Normalize text
# ---------------------------------------------------------

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ---------------------------------------------------------
# Detect label column
# ---------------------------------------------------------

if "label" not in df.columns:

    if "label_name" in df.columns:

        df["label"] = (
            df["label_name"]
            .map(
                {
                    "Human": 0,
                    "AI": 1,
                    "human": 0,
                    "ai": 1
                }
            )
        )

    else:

        raise ValueError(
            "Could not find label or label_name column."
        )


# ---------------------------------------------------------
# Detect model column
# ---------------------------------------------------------

if "model" not in df.columns:

    if "raid_model" in df.columns:

        df["model"] = df["raid_model"]

    else:

        raise ValueError(
            "Could not find RAID model column."
        )


# ---------------------------------------------------------
# Detect domain column
# ---------------------------------------------------------

if "domain" not in df.columns:

    df["domain"] = "unknown"


# ---------------------------------------------------------
# Remove invalid labels
# ---------------------------------------------------------

df = df[
    df["label"].isin([0, 1])
].copy()


df["label"] = (
    df["label"]
    .astype(int)
)


# ---------------------------------------------------------
# Length groups
# ---------------------------------------------------------

df["word_count"] = (
    df["text"]
    .apply(
        lambda x: len(x.split())
    )
)


df["length_group"] = (
    df["word_count"]
    .apply(get_length_group)
)


print(
    f"External samples: {len(df)}"
)


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print()
print("LABEL DISTRIBUTION")
print("==================")


if "label_name" in df.columns:

    print(
        df["label_name"]
        .value_counts()
        .to_string()
    )

else:

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


# =========================================================
# MODEL DISTRIBUTION
# =========================================================

print()
print("MODEL DISTRIBUTION")
print("==================")


print(
    df["model"]
    .value_counts()
    .to_string()
)


# =========================================================
# DOMAIN DISTRIBUTION
# =========================================================

print()
print("DOMAIN DISTRIBUTION")
print("===================")


print(
    df["domain"]
    .value_counts()
    .to_string()
)


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("TEXT LENGTH DISTRIBUTION")
print("========================")


print(
    pd.crosstab(
        df["length_group"],
        df["label"].map(
            {
                0: "Human",
                1: "AI"
            }
        )
    )
    .to_string()
)


# =========================================================
# LOAD TF-IDF MODELS
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
# WORD TF-IDF
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
# CHARACTER TF-IDF
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
# LOAD MINILM
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
# MINILM EMBEDDINGS
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


# ---------------------------------------------------------
# SCALE
# ---------------------------------------------------------

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
    f"Combined feature shape: "
    f"{X.shape}"
)


# =========================================================
# RUN PREDICTIONS
# =========================================================

print()
print("LOADING FINAL CLASSIFIER")
print("========================")


print(
    "Final classifier loaded."
)


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
    classifier
    .predict_proba(X)[:, 1]
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
print("FINAL RAID EXTERNAL RESULTS")
print("===========================")


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
# STORE PREDICTIONS
# =========================================================

results = df.copy()


results["prediction"] = y_pred


results["prediction_name"] = (
    results["prediction"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


results["ai_probability"] = (
    y_probability
)


results["human_probability"] = (
    1.0 - y_probability
)


# =========================================================
# PERFORMANCE BY GENERATOR
# =========================================================

print()
print("PERFORMANCE BY RAID GENERATOR")
print("=============================")


for model in sorted(
    results["model"].unique()
):

    subset = results[
        results["model"] == model
    ]


    model_accuracy = accuracy_score(
        subset["label"],
        subset["prediction"]
    )


    model_f1 = f1_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    print(
        f"{model:12s} "
        f"Accuracy={model_accuracy:.4f} "
        f"F1={model_f1:.4f} "
        f"Samples={len(subset)}"
    )


# =========================================================
# AI RECALL BY GENERATOR
# =========================================================

print()
print("AI DETECTION BY GENERATOR")
print("=========================")


for model in sorted(
    results["model"].unique()
):

    subset = results[
        (
            results["model"] == model
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
        f"{model:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"Samples={len(subset)}"
    )


# =========================================================
# PERFORMANCE BY DOMAIN
# =========================================================

print()
print("PERFORMANCE BY DOMAIN")
print("=====================")


for domain in sorted(
    results["domain"].unique()
):

    subset = results[
        results["domain"] == domain
    ]


    domain_accuracy = accuracy_score(
        subset["label"],
        subset["prediction"]
    )


    domain_f1 = f1_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    print(
        f"{domain:12s} "
        f"Accuracy={domain_accuracy:.4f} "
        f"F1={domain_f1:.4f} "
        f"Samples={len(subset)}"
    )


# =========================================================
# AI RECALL BY DOMAIN
# =========================================================

print()
print("AI DETECTION BY DOMAIN")
print("======================")


for domain in sorted(
    results["domain"].unique()
):

    subset = results[
        (
            results["domain"] == domain
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
        f"{domain:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={len(subset)}"
    )


# =========================================================
# PERFORMANCE BY TEXT LENGTH
# =========================================================

print()
print("PERFORMANCE BY TEXT LENGTH")
print("===========================")


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
# AI RECALL BY TEXT LENGTH
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

results.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("DETAILED RESULTS SAVED")
print("======================")


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
print("FINAL RAID SUMMARY")
print("==================")


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
    "FINAL RAID EVALUATION COMPLETED"
)