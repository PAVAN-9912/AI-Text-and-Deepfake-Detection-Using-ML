import os
import joblib
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = r"models\text_minilm_balanced"

DATA_FILE = (
    r"datasets\text\RAID\raid_external.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("RAID EXTERNAL TEST — BALANCED MINILM")
print("====================================")


# ============================================================
# LOAD RAID DATA
# ============================================================

print()
print("LOADING RAID DATA")
print("=================")


if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"RAID dataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
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
# LOAD MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


print(
    "Transformer:",
    TRANSFORMER_NAME
)

print(
    "Device:",
    device
)


embedding_model = SentenceTransformer(
    TRANSFORMER_NAME,
    device=device
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# LOAD SAVED CLASSIFIER
# ============================================================

print()
print("LOADING BALANCED CLASSIFIER")
print("===========================")


classifier_path = os.path.join(
    MODEL_DIR,
    "text_classifier.pkl"
)


scaler_path = os.path.join(
    MODEL_DIR,
    "linguistic_scaler.pkl"
)


if not os.path.exists(
    classifier_path
):

    raise FileNotFoundError(
        f"Classifier not found:\n{classifier_path}"
    )


if not os.path.exists(
    scaler_path
):

    raise FileNotFoundError(
        f"Linguistic scaler not found:\n{scaler_path}"
    )


classifier = joblib.load(
    classifier_path
)


linguistic_scaler = joblib.load(
    scaler_path
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

    text = str(text)

    words = text.split()

    word_count = len(
        words
    )


    # --------------------------------------------------------
    # Sentence detection
    # --------------------------------------------------------

    sentences = []

    current = ""


    for char in text:

        current += char

        if char in ".!?":

            if current.strip():

                sentences.append(
                    current.strip()
                )

            current = ""


    if current.strip():

        sentences.append(
            current.strip()
        )


    sentence_count = max(
        len(sentences),
        1
    )


    # --------------------------------------------------------
    # Average sentence length
    # --------------------------------------------------------

    avg_sentence_length = (
        word_count /
        sentence_count
    )


    # --------------------------------------------------------
    # Vocabulary diversity
    # --------------------------------------------------------

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


# ============================================================
# BUILD LINGUISTIC FEATURES
# ============================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


linguistic = np.array(
    [
        linguistic_features(text)
        for text in texts
    ],
    dtype=np.float32
)


linguistic = (
    linguistic_scaler.transform(
        linguistic
    )
)


print(
    "Linguistic feature shape:",
    linguistic.shape
)


# ============================================================
# CREATE MINILM EMBEDDINGS
# ============================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


embeddings = (
    embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
)


print()
print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X = np.hstack(
    [
        embeddings,
        linguistic
    ]
)


print()
print(
    "Combined feature shape:",
    X.shape
)


# ============================================================
# TRUE LABELS
# ============================================================

y_true = (
    df["label"]
    .astype(int)
    .values
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


predictions = (
    classifier.predict(
        X
    )
)


probabilities = (
    classifier.predict_proba(
        X
    )[:, 1]
)


# ============================================================
# OVERALL METRICS
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


# ============================================================
# RESULTS
# ============================================================

print()
print("RAID BALANCED MINILM RESULTS")
print("============================")


print(
    "Accuracy :",
    f"{accuracy:.4f}"
)


print(
    "Precision:",
    f"{precision:.4f}"
)


print(
    "Recall   :",
    f"{recall:.4f}"
)


print(
    "F1 Score :",
    f"{f1:.4f}"
)


print(
    "ROC-AUC  :",
    f"{roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")


print(
    confusion_matrix(
        y_true,
        predictions
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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
# PERFORMANCE BY RAID SOURCE
# ============================================================

print()
print("PERFORMANCE BY RAID MODEL")
print("=========================")


for model_name in sorted(
    df["model"].unique()
):

    mask = (
        df["model"] == model_name
    )


    model_true = y_true[
        mask
    ]


    model_pred = predictions[
        mask
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
# AI GENERATOR RECALL
# ============================================================

print()
print("AI DETECTION BY GENERATOR")
print("=========================")


for model_name in sorted(
    df["model"].unique()
):

    if model_name == "human":

        continue


    mask = (
        df["model"] == model_name
    )


    model_true = y_true[
        mask
    ]


    model_pred = predictions[
        mask
    ]


    generator_recall = (
        recall_score(
            model_true,
            model_pred,
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

results = df.copy()


results["prediction"] = (
    predictions
)


results["ai_probability"] = (
    probabilities
)


results["human_probability"] = (
    1.0 - probabilities
)


results["prediction_name"] = (
    results["prediction"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


output_file = (
    r"datasets\text\RAID"
    r"\raid_minilm_balanced_results.csv"
)


results.to_csv(
    output_file,
    index=False
)


print()
print("DETAILED RESULTS SAVED")
print("======================")

print(
    output_file
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("FINAL RAID SUMMARY")
print("==================")


print(
    f"Total samples       : {len(df)}"
)

print(
    f"Correct predictions : "
    f"{int((predictions == y_true).sum())}"
)

print(
    f"Incorrect predictions: "
    f"{int((predictions != y_true).sum())}"
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
    "RAID BALANCED MINILM "
    "EVALUATION COMPLETED"
)