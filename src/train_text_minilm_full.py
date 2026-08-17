import os
import re
import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

TRAIN_FILE = (
    r"datasets\text\HC3\train.csv"
)

VALIDATION_FILE = (
    r"datasets\text\HC3\validation.csv"
)

MODEL_DIR = (
    r"models\text_minilm_full"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_LENGTH = 256

BATCH_SIZE = 16


# ============================================================
# HEADER
# ============================================================

print()
print("FULL MINILM + LINGUISTIC TEXT MODEL")
print("===================================")


print(
    "Transformer:",
    TRANSFORMER_NAME
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


train_df["text"] = (
    train_df["text"]
    .fillna("")
    .astype(str)
)

validation_df["text"] = (
    validation_df["text"]
    .fillna("")
    .astype(str)
)


train_df["label"] = (
    train_df["label"]
    .astype(int)
)

validation_df["label"] = (
    validation_df["label"]
    .astype(int)
)


print(
    "Training samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(validation_df)
)


print()
print("TRAIN LABEL DISTRIBUTION")
print("=========================")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


print()
print("VALIDATION LABEL DISTRIBUTION")
print("=============================")

print(
    validation_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# BASIC LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

    text = str(text)

    words = re.findall(
        r"\b\w+\b",
        text
    )

    word_count = len(words)


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


# ============================================================
# BUILD LINGUISTIC FEATURES
# ============================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


train_linguistic = np.array(
    [
        linguistic_features(text)
        for text in train_df["text"]
    ],
    dtype=np.float32
)


validation_linguistic = np.array(
    [
        linguistic_features(text)
        for text in validation_df["text"]
    ],
    dtype=np.float32
)


print(
    "Training linguistic shape:",
    train_linguistic.shape
)

print(
    "Validation linguistic shape:",
    validation_linguistic.shape
)


# ============================================================
# STANDARDIZE LINGUISTIC FEATURES
# ============================================================

linguistic_scaler = StandardScaler()


train_linguistic = (
    linguistic_scaler.fit_transform(
        train_linguistic
    )
)


validation_linguistic = (
    linguistic_scaler.transform(
        validation_linguistic
    )
)


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


tokenizer = (
    AutoTokenizer.from_pretrained(
        TRANSFORMER_NAME
    )
)


encoder = (
    AutoModel.from_pretrained(
        TRANSFORMER_NAME
    )
)


encoder.to(device)

encoder.eval()


print(
    "MiniLM loaded successfully."
)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_embeddings(
    texts,
    dataset_name
):

    all_embeddings = []

    total = len(texts)


    print()
    print(
        "CREATING",
        dataset_name.upper(),
        "EMBEDDINGS"
    )

    print(
        "Total:",
        total
    )


    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch_texts = texts[
            start:
            start + BATCH_SIZE
        ]


        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )


        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }


        with torch.no_grad():

            outputs = encoder(
                **encoded
            )


        token_embeddings = (
            outputs.last_hidden_state
        )


        attention_mask = (
            encoded["attention_mask"]
        )


        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                token_embeddings.size()
            )
            .float()
        )


        summed = torch.sum(
            token_embeddings * mask,
            dim=1
        )


        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )


        embeddings = (
            summed / counts
        )


        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )


        all_embeddings.append(
            embeddings
            .cpu()
            .numpy()
        )


        processed = min(
            start + BATCH_SIZE,
            total
        )


        if (
            processed % 320 == 0
            or processed == total
        ):

            print(
                f"Processed "
                f"{processed}/{total}"
            )


    return np.vstack(
        all_embeddings
    ).astype(
        np.float32
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

train_embeddings = create_embeddings(
    train_df["text"].tolist(),
    "training"
)


validation_embeddings = create_embeddings(
    validation_df["text"].tolist(),
    "validation"
)


# ============================================================
# EMBEDDING SHAPES
# ============================================================

print()
print("EMBEDDING SHAPES")
print("================")


print(
    "Training:",
    train_embeddings.shape
)

print(
    "Validation:",
    validation_embeddings.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


X_train = np.hstack(
    [
        train_embeddings,
        train_linguistic
    ]
)


X_validation = np.hstack(
    [
        validation_embeddings,
        validation_linguistic
    ]
)


print(
    "Combined training shape:",
    X_train.shape
)

print(
    "Combined validation shape:",
    X_validation.shape
)


# ============================================================
# LABELS
# ============================================================

y_train = (
    train_df["label"]
    .values
)

y_validation = (
    validation_df["label"]
    .values
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("TRAINING LOGISTIC REGRESSION")
print("============================")


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    random_state=42
)


classifier.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

print()
print("EVALUATING VALIDATION SET")
print("==========================")


predictions = (
    classifier.predict(
        X_validation
    )
)


probabilities = (
    classifier.predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_validation,
    predictions
)


precision = precision_score(
    y_validation,
    predictions,
    zero_division=0
)


recall = recall_score(
    y_validation,
    predictions,
    zero_division=0
)


f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0
)


auc = roc_auc_score(
    y_validation,
    probabilities
)


# ============================================================
# RESULTS
# ============================================================

print()
print("FULL MINILM MODEL RESULTS")
print("==========================")


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
    f"ROC-AUC  : {auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")


cm = confusion_matrix(
    y_validation,
    predictions
)


print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("CLASSIFICATION REPORT")
print("=====================")


print(
    classification_report(
        y_validation,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("SAVING MODEL")
print("============")


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


joblib.dump(
    {
        "model_type":
            "minilm_linguistic_logistic_regression",

        "transformer_name":
            TRANSFORMER_NAME,

        "embedding_dimension":
            int(
                train_embeddings.shape[1]
            ),

        "linguistic_features":
            [
                "word_count",
                "sentence_count",
                "avg_sentence_length",
                "vocabulary_diversity"
            ],

        "combined_dimension":
            int(
                X_train.shape[1]
            ),

        "max_length":
            MAX_LENGTH,

        "train_samples":
            len(train_df),

        "validation_samples":
            len(validation_df)
    },
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


joblib.dump(
    {
        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(auc)
    },
    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    )
)


print()
print("MODEL SAVED")
print("===========")

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


print()
print(
    "FULL MINILM MODEL TRAINING COMPLETED"
)