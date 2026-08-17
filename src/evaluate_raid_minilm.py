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
# PATHS
# ============================================================

MODEL_DIR = r"models\text_minilm_full"

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
print("RAID EXTERNAL TEST — FULL MINILM")
print("================================")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


print()
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


model = SentenceTransformer(
    TRANSFORMER_NAME,
    device=device
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print()
print("LOADING CLASSIFIER")
print("==================")

classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


linguistic_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

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
        len(sentences),
        1
    )

    avg_sentence_length = (
        word_count /
        sentence_count
    )

    unique_words = len(
        set(
            w.lower()
            for w in words
        )
    )

    vocabulary_diversity = (
        unique_words /
        max(word_count, 1)
    )

    return [
        word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity
    ]


# ============================================================
# BUILD LINGUISTIC MATRIX
# ============================================================

linguistic = np.array(
    [
        linguistic_features(text)
        for text in df["text"].fillna("")
    ],
    dtype=np.float32
)


linguistic = linguistic_scaler.transform(
    linguistic
)


print()
print(
    "Linguistic feature shape:",
    linguistic.shape
)


# ============================================================
# CREATE MINILM EMBEDDINGS
# ============================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("==========================")

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
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
# PREDICTION
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


y_true = (
    df["label"]
    .astype(int)
    .values
)


predictions = classifier.predict(
    X
)


probabilities = classifier.predict_proba(
    X
)[:, 1
]


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

auc = roc_auc_score(
    y_true,
    probabilities
)


# ============================================================
# RESULTS
# ============================================================

print()
print("RAID FULL MINILM RESULTS")
print("========================")

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
    f"{auc:.4f}"
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
# GENERATOR-WISE PERFORMANCE
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

    yt = y_true[mask]

    yp = predictions[mask]

    acc = accuracy_score(
        yt,
        yp
    )

    print(
        f"{model_name:12s} "
        f"Accuracy={acc:.4f} "
        f"Samples={mask.sum()}"
    )


# ============================================================
# GENERATOR-WISE AI RECALL
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

    yt = y_true[mask]

    yp = predictions[mask]

    generator_recall = recall_score(
        yt,
        yp,
        zero_division=0
    )

    print(
        f"{model_name:12s} "
        f"AI Recall={generator_recall:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = df.copy()

results["prediction"] = predictions

results["ai_probability"] = probabilities

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
    r"\raid_minilm_results.csv"
)


results.to_csv(
    output_file,
    index=False
)


print()
print("Detailed results saved:")
print(output_file)


print()
print("RAID FULL MINILM EVALUATION COMPLETED")