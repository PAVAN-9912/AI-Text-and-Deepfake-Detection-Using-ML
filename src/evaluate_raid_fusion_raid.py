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


# ============================================================
# CONFIGURATION
# ============================================================

RAID_EXTERNAL = r"datasets\text\RAID\raid_external.csv"

MODEL_DIR = r"models\text_fusion_raid"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BATCH_SIZE = 32


# ============================================================
# HEADER
# ============================================================

print()
print("FINAL RAID EXTERNAL TEST")
print("=========================")

print(
    "Model: HC3 + RAID Fusion"
)


# ============================================================
# LOAD RAID
# ============================================================

print()
print("LOADING RAID EXTERNAL DATA")
print("===========================")


df = pd.read_csv(
    RAID_EXTERNAL
)


df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)


df["label"] = (
    df["label"]
    .astype(int)
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
print("MODEL DISTRIBUTION")
print("==================")

print(
    df["model"].value_counts()
)


# ============================================================
# LOAD TF-IDF
# ============================================================

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

    word_vectorizer = pickle.load(
        f
    )


with open(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    char_vectorizer = pickle.load(
        f
    )


# ============================================================
# BUILD WORD FEATURES
# ============================================================

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


# ============================================================
# BUILD CHARACTER FEATURES
# ============================================================

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


transformer = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


X_embedding = (
    transformer.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
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


def linguistic_features(text):

    text = str(text)

    words = text.split()

    word_count = len(
        words
    )

    sentence_count = max(
        text.count(".")
        + text.count("!")
        + text.count("?"),
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


X_linguistic = np.array(
    [
        linguistic_features(x)
        for x in df["text"]
    ],
    dtype=np.float32
)


print(
    "Raw linguistic shape:",
    X_linguistic.shape
)


# ============================================================
# SCALE LINGUISTIC FEATURES
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    ),
    "rb"
) as f:

    scaler = pickle.load(
        f
    )


X_linguistic = (
    scaler.transform(
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


X = hstack(
    [
        X_word,

        X_char,

        csr_matrix(
            X_embedding
        ),

        csr_matrix(
            X_linguistic
        )
    ]
).tocsr()


print(
    "Combined feature shape:",
    X.shape
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

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

    classifier = pickle.load(
        f
    )


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


y_true = df[
    "label"
].values


y_pred = classifier.predict(
    X
)


y_probability = classifier.predict_proba(
    X
)[:, 1]


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")

print(
    confusion_matrix(
        y_true,
        y_pred
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
        y_pred,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# PERFORMANCE BY GENERATOR
# ============================================================

print()
print("PERFORMANCE BY RAID GENERATOR")
print("=============================")


for model in sorted(
    df["model"].unique()
):

    mask = (
        df["model"] == model
    )


    model_true = y_true[
        mask
    ]

    model_pred = y_pred[
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
        f"{model:12s} "
        f"Accuracy={model_accuracy:.4f} "
        f"F1={model_f1:.4f} "
        f"Samples={mask.sum()}"
    )


# ============================================================
# AI RECALL BY GENERATOR
# ============================================================

print()
print("AI DETECTION BY GENERATOR")
print("==========================")


for model in sorted(
    df[
        df["label"] == 1
    ]["model"].unique()
):

    mask = (
        (df["model"] == model)
        &
        (df["label"] == 1)
    )


    generator_true = y_true[
        mask
    ]

    generator_pred = y_pred[
        mask
    ]


    generator_recall = (
        recall_score(
            generator_true,
            generator_pred,
            zero_division=0
        )
    )


    print(
        f"{model:12s} "
        f"AI Recall={generator_recall:.4f}"
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

output_file = (
    r"datasets\text\RAID"
    r"\raid_fusion_raid_results.csv"
)


results = df.copy()


results["predicted_label"] = (
    y_pred
)


results["ai_probability"] = (
    y_probability
)


results["predicted_label_name"] = (
    np.where(
        y_pred == 1,
        "AI",
        "Human"
    )
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
    "Total samples       :",
    len(df)
)

print(
    "Correct predictions :",
    int(
        (y_true == y_pred).sum()
    )
)

print(
    "Incorrect predictions:",
    int(
        (y_true != y_pred).sum()
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
    "FINAL RAID EXTERNAL EVALUATION COMPLETED"
)