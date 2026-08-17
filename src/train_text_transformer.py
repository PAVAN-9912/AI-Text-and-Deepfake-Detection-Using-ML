import os
import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    AutoModel
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = r"datasets\text\HC3\train.csv"

VALIDATION_FILE = (
    r"datasets\text\HC3\validation.csv"
)

MODEL_DIR = (
    r"models\text_transformer"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Prototype size
# ------------------------------------------------------------

TRAIN_SAMPLES = 2000

VALIDATION_SAMPLES = 1000

BATCH_SIZE = 16

MAX_LENGTH = 256


# ------------------------------------------------------------
# Lightweight transformer
# ------------------------------------------------------------

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# HEADER
# ============================================================

print()
print("TRANSFORMER TEXT DETECTOR")
print("=========================")

print(
    "Prototype mode"
)

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


# ------------------------------------------------------------
# Shuffle before taking prototype subset
# ------------------------------------------------------------

train_df = (
    train_df
    .sample(
        n=min(
            TRAIN_SAMPLES,
            len(train_df)
        ),
        random_state=42
    )
    .reset_index(drop=True)
)


validation_df = (
    validation_df
    .sample(
        n=min(
            VALIDATION_SAMPLES,
            len(validation_df)
        ),
        random_state=42
    )
    .reset_index(drop=True)
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
# LOAD TRANSFORMER
# ============================================================

print()
print("LOADING TRANSFORMER")
print("===================")


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
    "Transformer loaded successfully."
)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_embeddings(
    texts,
    name
):

    embeddings = []


    total = len(texts)


    print()
    print(
        "Creating embeddings:",
        name
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


        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

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


        batch_embeddings = (
            summed / counts
        )


        batch_embeddings = (
            batch_embeddings
            .cpu()
            .numpy()
        )


        embeddings.append(
            batch_embeddings
        )


        processed = min(
            start + BATCH_SIZE,
            total
        )


        print(
            f"Processed "
            f"{processed}/{total}"
        )


    return np.vstack(
        embeddings
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

X_train = create_embeddings(
    train_df["text"].tolist(),
    "training"
)


X_validation = create_embeddings(
    validation_df["text"].tolist(),
    "validation"
)


# ============================================================
# SHAPES
# ============================================================

print()
print("EMBEDDING SHAPES")
print("================")


print(
    "Training:",
    X_train.shape
)


print(
    "Validation:",
    X_validation.shape
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("TRAINING CLASSIFIER")
print("===================")


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
    "Classifier training completed."
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("VALIDATION")
print("==========")


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
print("TRANSFORMER PROTOTYPE RESULTS")
print("=============================")


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


print(
    confusion_matrix(
        y_validation,
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
print("SAVING PROTOTYPE")
print("================")


joblib.dump(
    classifier,
    os.path.join(
        MODEL_DIR,
        "transformer_classifier.pkl"
    )
)


joblib.dump(
    {
        "transformer_name":
            TRANSFORMER_NAME,

        "max_length":
            MAX_LENGTH,

        "embedding_dimension":
            int(X_train.shape[1]),

        "train_samples":
            len(train_df),

        "validation_samples":
            len(validation_df)
    },
    os.path.join(
        MODEL_DIR,
        "transformer_config.pkl"
    )
)


metrics = {

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
}


joblib.dump(
    metrics,
    os.path.join(
        MODEL_DIR,
        "transformer_metrics.pkl"
    )
)


# ------------------------------------------------------------
# Save encoder information
# ------------------------------------------------------------

joblib.dump(
    TRANSFORMER_NAME,
    os.path.join(
        MODEL_DIR,
        "transformer_name.pkl"
    )
)


print()
print(
    "Prototype saved to:",
    MODEL_DIR
)


print()
print(
    "TRANSFORMER PROTOTYPE COMPLETED"
)