import os
import re
import joblib
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


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"datasets\text\RAID\raid_external.csv"

MODEL_DIR = r"models\text_fusion"

MINILM_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# HEADER
# ============================================================

print()
print("RAID EXTERNAL TEST — TEXT FUSION MODEL")
print("=======================================")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "linguistic_scaler.pkl",
    "text_classifier.pkl",
    "text_config.pkl"
]


for filename in required_files:

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing model file:\n{path}"
        )


if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nMissing RAID dataset:\n{DATA_PATH}"
    )


# ============================================================
# LOAD RAID DATA
# ============================================================

print()
print("LOADING RAID DATA")
print("=================")


df = pd.read_csv(
    DATA_PATH
)


print(
    "External samples:",
    len(df)
)


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label_name"].value_counts()
)


print()
print("RAID MODEL DISTRIBUTION")
print("=======================")

print(
    df["model"].value_counts()
)


# ============================================================
# PREPARE TEXT
# ============================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


texts = (
    df["text"]
    .fillna("")
    .map(clean_text)
    .tolist()
)


# ============================================================
# LABELS
# ============================================================

y_true = (
    df["label"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# LOAD TF-IDF MODELS
# ============================================================

print()
print("LOADING TF-IDF MODELS")
print("====================")


word_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)


char_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)


# ============================================================
# WORD TF-IDF
# ============================================================

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


# ============================================================
# CHARACTER TF-IDF
# ============================================================

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


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


print(
    "Transformer:",
    MINILM_NAME
)


minilm = SentenceTransformer(
    MINILM_NAME
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# CREATE MINILM EMBEDDINGS
# ============================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


X_minilm = minilm.encode(

    texts,

    batch_size=16,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


print(
    "MiniLM embedding shape:",
    X_minilm.shape
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

    text = str(text)


    words = re.findall(
        r"\b\w+\b",
        text
    )


    word_count = len(
        words
    )


    sentence_count = max(
        len(
            re.findall(
                r"[.!?]+",
                text
            )
        ),
        1
    )


    avg_sentence_length = (
        word_count /
        sentence_count
    )


    unique_words = len(
        set(
            word.lower()
            for word in words
        )
    )


    vocabulary_diversity = (
        unique_words /
        max(
            word_count,
            1
        )
    )


    return [

        word_count,

        sentence_count,

        avg_sentence_length,

        vocabulary_diversity

    ]


print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_linguistic = np.array(
    [
        linguistic_features(text)
        for text in texts
    ],
    dtype=float
)


print(
    "Raw linguistic shape:",
    X_linguistic.shape
)


# ============================================================
# LOAD SCALER
# ============================================================

linguistic_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic
    )
)


print(
    "Scaled linguistic shape:",
    X_linguistic.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


X_dense = np.hstack(
    [
        X_minilm,
        X_linguistic
    ]
)


X_combined = hstack(
    [
        X_word,
        X_char,
        csr_matrix(
            X_dense
        )
    ]
).tocsr()


print(
    "Combined feature shape:",
    X_combined.shape
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print()
print("LOADING FUSION CLASSIFIER")
print("=========================")


classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


predictions = classifier.predict(
    X_combined
)


probabilities = (
    classifier.predict_proba(
        X_combined
    )[:, 1]
)


# ============================================================
# METRICS
# ============================================================

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


cm = confusion_matrix(
    y_true,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("RAID TEXT FUSION RESULTS")
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


print()
print("CONFUSION MATRIX")
print("================")

print(
    cm
)


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


# ============================================================
# PERFORMANCE BY RAID GENERATOR
# ============================================================

print()
print("PERFORMANCE BY RAID MODEL")
print("=========================")


for model_name in sorted(
    df["model"].unique()
):

    mask = (
        df["model"] ==
        model_name
    )


    model_true = y_true[
        mask.to_numpy()
    ]


    model_pred = predictions[
        mask.to_numpy()
    ]


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
        f"{model_name:12s} "
        f"Accuracy={model_accuracy:.4f} "
        f"F1={model_f1:.4f} "
        f"Samples={mask.sum()}"
    )


# ============================================================
# AI DETECTION BY GENERATOR
# ============================================================

print()
print("AI DETECTION BY GENERATOR")
print("==========================")


ai_df = df[
    df["label"] == 1
]


for model_name in sorted(
    ai_df["model"].unique()
):

    mask = (
        df["model"] ==
        model_name
    ) & (
        df["label"] ==
        1
    )


    generator_true = y_true[
        mask.to_numpy()
    ]


    generator_pred = predictions[
        mask.to_numpy()
    ]


    generator_recall = (
        recall_score(
            generator_true,
            generator_pred,
            zero_division=0
        )
    )


    print(
        f"{model_name:12s} "
        f"AI Recall={generator_recall:.4f}"
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df = df.copy()


results_df[
    "prediction"
] = predictions


results_df[
    "ai_probability"
] = probabilities


results_df[
    "human_probability"
] = 1.0 - probabilities


results_df[
    "correct"
] = (
    predictions ==
    y_true
)


output_path = (
    r"datasets\text\RAID"
    r"\raid_fusion_results.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print()
print("DETAILED RESULTS SAVED")
print("======================")

print(
    output_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("FINAL RAID FUSION SUMMARY")
print("=========================")


print(
    "Total samples       :",
    len(df)
)


print(
    "Correct predictions :",
    int(
        np.sum(
            predictions ==
            y_true
        )
    )
)


print(
    "Incorrect predictions:",
    int(
        np.sum(
            predictions !=
            y_true
        )
    )
)


print(
    f"Accuracy            : {accuracy:.4f}"
)


print(
    f"F1 Score            : {f1:.4f}"
)


print(
    f"ROC-AUC             : {roc_auc:.4f}"
)


print()
print(
    "RAID TEXT FUSION EVALUATION COMPLETED"
)