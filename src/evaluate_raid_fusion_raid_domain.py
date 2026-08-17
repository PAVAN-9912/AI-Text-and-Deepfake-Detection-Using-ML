import os
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

RAID_EXTERNAL = (
    "datasets\\text\\RAID\\raid_external.csv"
)

MODEL_DIR = (
    "models\\text_fusion_raid_domain"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OUTPUT_PATH = (
    "datasets\\text\\RAID\\"
    "raid_fusion_raid_domain_results.csv"
)


# =========================================================
# HEADER
# =========================================================

print()
print("FINAL RAID EXTERNAL TEST — DOMAIN-AWARE FUSION")
print("===============================================")


# =========================================================
# LOAD RAID EXTERNAL DATA
# =========================================================

print()
print("LOADING RAID EXTERNAL DATA")
print("===========================")


df = pd.read_csv(
    RAID_EXTERNAL
)


print(
    f"External samples: {len(df)}"
)


# ---------------------------------------------------------
# Ensure label column exists
# ---------------------------------------------------------

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
            "No label or label_name column found."
        )


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


if "model" in df.columns:

    model_column = "model"

elif "raid_model" in df.columns:

    model_column = "raid_model"

else:

    model_column = None


if model_column:

    print(
        df[model_column]
        .value_counts()
        .to_string()
    )


# =========================================================
# DOMAIN DISTRIBUTION
# =========================================================

if "domain" in df.columns:

    print()
    print("DOMAIN DISTRIBUTION")
    print("===================")

    print(
        df["domain"]
        .value_counts()
        .to_string()
    )


# =========================================================
# PREPARE TEXT
# =========================================================

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


y_true = (
    df["label"]
    .astype(int)
    .values
)


# =========================================================
# LOAD TF-IDF MODELS
# =========================================================

print()
print("LOADING TF-IDF MODELS")
print("=====================")


with open(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    word_vectorizer = pickle.load(f)


with open(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    char_vectorizer = pickle.load(f)


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = (
    word_vectorizer.transform(
        texts
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
print("===========================")


X_char = (
    char_vectorizer.transform(
        texts
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# =========================================================
# LOAD MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")


transformer = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# =========================================================
# CREATE MINILM EMBEDDINGS
# =========================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


X_embed = transformer.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


print(
    "Embedding shape:",
    X_embed.shape
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


def linguistic_features(texts):

    features = []


    for text in texts:

        text = str(text)

        words = text.split()

        word_count = len(words)


        sentences = [
            s.strip()
            for s in text.replace(
                "!",
                "."
            ).replace(
                "?",
                "."
            ).split(".")
            if s.strip()
        ]


        sentence_count = max(
            1,
            len(sentences)
        )


        avg_sentence_length = (
            word_count /
            sentence_count
        )


        if word_count > 0:

            vocabulary_diversity = (
                len(
                    set(
                        w.lower()
                        for w in words
                    )
                )
                /
                word_count
            )

        else:

            vocabulary_diversity = 0.0


        avg_word_length = (
            sum(
                len(w)
                for w in words
            )
            /
            max(
                1,
                word_count
            )
        )


        features.append(
            [
                word_count,
                sentence_count,
                avg_sentence_length,
                vocabulary_diversity,
                avg_word_length,
            ]
        )


    return np.asarray(
        features,
        dtype=np.float32
    )


X_ling_raw = linguistic_features(
    texts
)


print(
    "Raw linguistic shape:",
    X_ling_raw.shape
)


# =========================================================
# LOAD LINGUISTIC SCALER
# =========================================================

with open(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    ),
    "rb"
) as f:

    linguistic_scaler = pickle.load(f)


X_ling = (
    linguistic_scaler.transform(
        X_ling_raw
    )
)


print(
    "Scaled linguistic shape:",
    X_ling.shape
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
        X_embed,
        X_ling,
    ],
    format="csr"
)


print(
    "Combined feature shape:",
    X.shape
)


# =========================================================
# LOAD CLASSIFIER
# =========================================================

print()
print("LOADING FINAL CLASSIFIER")
print("========================")


with open(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    ),
    "rb"
) as f:

    classifier = pickle.load(f)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


predictions = classifier.predict(
    X
)


probabilities = classifier.predict_proba(
    X
)[:, 1]


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


cm = confusion_matrix(
    y_true,
    predictions
)


print(cm)


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
# PERFORMANCE BY GENERATOR
# =========================================================

if model_column:

    print()
    print("PERFORMANCE BY RAID GENERATOR")
    print("=============================")


    for model in sorted(
        df[model_column]
        .dropna()
        .unique()
    ):

        mask = (
            df[model_column]
            == model
        )


        model_true = y_true[mask]

        model_pred = predictions[mask]


        model_accuracy = (
            accuracy_score(
                model_true,
                model_pred
            )
        )


        model_f1 = (
            f1_score(
                model_true,
                model_pred,
                zero_division=0
            )
        )


        print(
            f"{model:12s} "
            f"Accuracy={model_accuracy:.4f} "
            f"F1={model_f1:.4f} "
            f"Samples={mask.sum()}"
        )


# =========================================================
# AI DETECTION BY GENERATOR
# =========================================================

if model_column:

    print()
    print("AI DETECTION BY GENERATOR")
    print("=========================")


    ai_models = [
        "gpt2",
        "llama-chat",
        "mpt",
        "mpt-chat",
    ]


    for model in ai_models:

        mask = (
            (df[model_column] == model)
            &
            (y_true == 1)
        )


        if mask.sum() == 0:

            continue


        ai_recall = (
            recall_score(
                y_true[mask],
                predictions[mask],
                zero_division=0
            )
        )


        print(
            f"{model:12s} "
            f"AI Recall={ai_recall:.4f}"
        )


# =========================================================
# PERFORMANCE BY DOMAIN
# =========================================================

if "domain" in df.columns:

    print()
    print("PERFORMANCE BY DOMAIN")
    print("=====================")


    for domain in sorted(
        df["domain"]
        .dropna()
        .unique()
    ):

        mask = (
            df["domain"] == domain
        )


        domain_accuracy = (
            accuracy_score(
                y_true[mask],
                predictions[mask]
            )
        )


        domain_f1 = (
            f1_score(
                y_true[mask],
                predictions[mask],
                zero_division=0
            )
        )


        print(
            f"{domain:12s} "
            f"Accuracy={domain_accuracy:.4f} "
            f"F1={domain_f1:.4f} "
            f"Samples={mask.sum()}"
        )


# =========================================================
# SAVE DETAILED RESULTS
# =========================================================

results = df.copy()


results["predicted_label"] = (
    predictions
)

results["predicted_label_name"] = (
    np.where(
        predictions == 1,
        "AI",
        "Human"
    )
)

results["ai_probability"] = (
    probabilities
)

results["human_probability"] = (
    1.0 - probabilities
)

results["correct"] = (
    predictions == y_true
)


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

print()
print("FINAL RAID SUMMARY")
print("==================")


print(
    f"Total samples        : {len(df)}"
)

print(
    f"Correct predictions  : "
    f"{int((predictions == y_true).sum())}"
)

print(
    f"Incorrect predictions: "
    f"{int((predictions != y_true).sum())}"
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
    "FINAL RAID DOMAIN-AWARE "
    "EXTERNAL EVALUATION COMPLETED"
)