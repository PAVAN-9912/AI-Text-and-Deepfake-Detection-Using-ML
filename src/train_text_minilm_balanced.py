import os
import joblib
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

TRAIN_FILE = r"datasets\text\HC3\train.csv"
VALIDATION_FILE = r"datasets\text\HC3\validation.csv"

MODEL_DIR = r"models\text_minilm_balanced"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RANDOM_STATE = 42

# Maximum number of samples from EACH label
# within EACH source.
MAX_PER_SOURCE_LABEL = 5000


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("BALANCED MINILM TEXT DETECTOR")
print("=============================")

print(
    "Transformer:",
    TRANSFORMER_NAME
)

print(
    "Device:",
    device
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("LOADING DATA")
print("============")


train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)


print(
    "Original training samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(validation_df)
)


# ============================================================
# SOURCE DISTRIBUTION BEFORE BALANCING
# ============================================================

print()
print("ORIGINAL TRAIN DISTRIBUTION")
print("===========================")

print(
    train_df.groupby(
        ["source", "label"]
    ).size()
    .unstack(fill_value=0)
    .to_string()
)


# ============================================================
# SOURCE-BALANCED TRAINING DATA
# ============================================================

print()
print("BUILDING SOURCE-BALANCED TRAINING SET")
print("======================================")


balanced_parts = []


for source in sorted(
    train_df["source"].unique()
):

    source_df = train_df[
        train_df["source"] == source
    ].copy()


    human = source_df[
        source_df["label"] == 0
    ]


    ai = source_df[
        source_df["label"] == 1
    ]


    n = min(
        len(human),
        len(ai),
        MAX_PER_SOURCE_LABEL
    )


    if n == 0:

        print(
            f"Skipping {source}: "
            "missing one class."
        )

        continue


    human_sample = human.sample(
        n=n,
        random_state=RANDOM_STATE
    )


    ai_sample = ai.sample(
        n=n,
        random_state=RANDOM_STATE
    )


    balanced_parts.append(
        human_sample
    )

    balanced_parts.append(
        ai_sample
    )


    print(
        f"{source:15s} "
        f"Human={n:5d} "
        f"AI={n:5d}"
    )


# ============================================================
# COMBINE
# ============================================================

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)


balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


print()
print(
    "Balanced training samples:",
    len(balanced_df)
)


print()
print("BALANCED TRAIN DISTRIBUTION")
print("===========================")

print(
    balanced_df.groupby(
        ["source", "label"]
    ).size()
    .unstack(fill_value=0)
    .to_string()
)


print()
print("TOTAL LABEL DISTRIBUTION")
print("=========================")

print(
    balanced_df["label"].value_counts()
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


# ============================================================
# BUILD LINGUISTIC MATRICES
# ============================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_train_ling = np.array(
    [
        linguistic_features(text)
        for text in balanced_df["text"]
    ],
    dtype=np.float32
)


X_val_ling = np.array(
    [
        linguistic_features(text)
        for text in validation_df["text"]
    ],
    dtype=np.float32
)


print(
    "Training linguistic shape:",
    X_train_ling.shape
)

print(
    "Validation linguistic shape:",
    X_val_ling.shape
)


# ============================================================
# SCALE LINGUISTIC FEATURES
# ============================================================

linguistic_scaler = StandardScaler()


X_train_ling = (
    linguistic_scaler.fit_transform(
        X_train_ling
    )
)


X_val_ling = (
    linguistic_scaler.transform(
        X_val_ling
    )
)


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


embedding_model = SentenceTransformer(
    TRANSFORMER_NAME,
    device=device
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# CREATE TRAINING EMBEDDINGS
# ============================================================

print()
print("CREATING TRAINING EMBEDDINGS")
print("============================")

train_texts = (
    balanced_df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


X_train_embedding = (
    embedding_model.encode(
        train_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
)


print()
print(
    "Training embedding shape:",
    X_train_embedding.shape
)


# ============================================================
# CREATE VALIDATION EMBEDDINGS
# ============================================================

print()
print("CREATING VALIDATION EMBEDDINGS")
print("==============================")

validation_texts = (
    validation_df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


X_val_embedding = (
    embedding_model.encode(
        validation_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
)


print()
print(
    "Validation embedding shape:",
    X_val_embedding.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


X_train = np.hstack(
    [
        X_train_embedding,
        X_train_ling
    ]
)


X_val = np.hstack(
    [
        X_val_embedding,
        X_val_ling
    ]
)


print(
    "Combined training shape:",
    X_train.shape
)

print(
    "Combined validation shape:",
    X_val.shape
)


# ============================================================
# LABELS
# ============================================================

y_train = (
    balanced_df["label"]
    .astype(int)
    .values
)


y_val = (
    validation_df["label"]
    .astype(int)
    .values
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("TRAINING LOGISTIC REGRESSION")
print("============================")


classifier = LogisticRegression(
    max_iter=2000,
    class_weight=None,
    random_state=RANDOM_STATE
)


classifier.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print()
print("EVALUATING VALIDATION SET")
print("==========================")


val_predictions = classifier.predict(
    X_val
)


val_probabilities = (
    classifier.predict_proba(
        X_val
    )[:, 1]
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    val_predictions
)


precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)


recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)


f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_val,
    val_probabilities
)


# ============================================================
# RESULTS
# ============================================================

print()
print("BALANCED MINILM RESULTS")
print("=======================")

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
        y_val,
        val_predictions
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
        y_val,
        val_predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# SOURCE-WISE VALIDATION RESULTS
# ============================================================

print()
print("VALIDATION PERFORMANCE BY SOURCE")
print("================================")


for source in sorted(
    validation_df["source"].unique()
):

    mask = (
        validation_df["source"]
        == source
    )


    source_true = y_val[
        mask
    ]


    source_pred = val_predictions[
        mask
    ]


    source_accuracy = (
        accuracy_score(
            source_true,
            source_pred
        )
    )


    source_f1 = (
        f1_score(
            source_true,
            source_pred,
            zero_division=0
        )
    )


    print(
        f"{source:15s} "
        f"Accuracy={source_accuracy:.4f} "
        f"F1={source_f1:.4f} "
        f"Samples={mask.sum()}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("SAVING BALANCED MODEL")
print("=====================")


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    classifier,
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


joblib.dump(
    linguistic_scaler,
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)


config = {

    "transformer":
        TRANSFORMER_NAME,

    "embedding_dimension":
        384,

    "linguistic_features":
        [
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "vocabulary_diversity"
        ],

    "feature_dimension":
        388,

    "training_method":
        "source_balanced_hc3",

    "max_per_source_label":
        MAX_PER_SOURCE_LABEL,

    "random_state":
        RANDOM_STATE
}


joblib.dump(
    config,
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


metrics = {

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1":
        f1,

    "roc_auc":
        roc_auc,

    "training_samples":
        len(balanced_df),

    "validation_samples":
        len(validation_df)
}


joblib.dump(
    metrics,
    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    )
)


# ============================================================
# SAVE TRAINING DISTRIBUTION
# ============================================================

balanced_df.groupby(
    ["source", "label"]
).size().reset_index(
    name="count"
).to_csv(
    os.path.join(
        MODEL_DIR,
        "training_distribution.csv"
    ),
    index=False
)


print()
print("BALANCED MODEL SAVED")
print("====================")

print(
    "Directory:",
    MODEL_DIR
)

print(
    "Files:"
)

print(
    " - text_classifier.pkl"
)

print(
    " - linguistic_scaler.pkl"
)

print(
    " - text_config.pkl"
)

print(
    " - text_metrics.pkl"
)

print(
    " - training_distribution.csv"
)


print()
print(
    "BALANCED MINILM TRAINING COMPLETED"
)