import os
import copy
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = r"datasets\video\temporal_resnet_features.csv"

MODEL_DIR = r"models\video"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "temporal_attention_model.pth"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "video_scaler.pkl"
)

CONFIG_FILE = os.path.join(
    MODEL_DIR,
    "video_config.pkl"
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "video_metrics.pkl"
)


SEQUENCE_LENGTH = 8
FEATURE_DIM = 512

HIDDEN_DIM = 128

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

RANDOM_SEED = 42

THRESHOLD = 0.50


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("FINAL VIDEO MODEL TRAINING")
print("==========================")
print("Device:", device)
print("Sequence length:", SEQUENCE_LENGTH)
print("Feature dimension:", FEATURE_DIM)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLOADING TEMPORAL RESNET FEATURES")
print("================================")

df = pd.read_csv(FEATURE_FILE)

print("Total samples:", len(df))


# ============================================================
# FIND FEATURE COLUMNS
# ============================================================

metadata_columns = {
    "video_id",
    "split",
    "label",
    "label_name",
    "filename",
    "frame_count",
    "frame_directory"
}


feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


expected_features = (
    SEQUENCE_LENGTH * FEATURE_DIM
)


print(
    "Temporal feature columns:",
    len(feature_columns)
)

print(
    "Expected:",
    expected_features
)


assert len(feature_columns) == expected_features, (
    f"Expected {expected_features} feature columns, "
    f"found {len(feature_columns)}"
)


# ============================================================
# CHECK VIDEO IDs
# ============================================================

assert (
    df["video_id"].nunique()
    == len(df)
), "Duplicate video IDs detected!"


print(
    "Unique video IDs:",
    df["video_id"].nunique()
)


# ============================================================
# CREATE TEMPORAL FEATURE MATRIX
# ============================================================

X_flat = df[
    feature_columns
].values.astype(
    np.float32
)


X = X_flat.reshape(
    -1,
    SEQUENCE_LENGTH,
    FEATURE_DIM
)


y = df[
    "label"
].values.astype(
    np.float32
)


print("\nFEATURE MATRIX")
print("==============")

print(
    "Flat:",
    X_flat.shape
)

print(
    "Temporal:",
    X.shape
)


# ============================================================
# DATASET SPLIT
# ============================================================

train_mask = (
    df["split"].values == "train"
)

val_mask = (
    df["split"].values == "validation"
)

test_mask = (
    df["split"].values == "test"
)


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nDATASET SPLIT")
print("=============")

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTRAIN LABEL DISTRIBUTION")
print(
    pd.Series(
        y_train
    ).value_counts().sort_index()
)

print("\nVALIDATION LABEL DISTRIBUTION")
print(
    pd.Series(
        y_val
    ).value_counts().sort_index()
)

print("\nTEST LABEL DISTRIBUTION")
print(
    pd.Series(
        y_test
    ).value_counts().sort_index()
)


# ============================================================
# STANDARDIZATION
# ============================================================
# IMPORTANT:
# Mean and standard deviation are calculated ONLY from
# training data, exactly like the original model.

print("\nSTANDARDIZATION")
print("================")

train_flat = X_train.reshape(
    -1,
    FEATURE_DIM
)


feature_mean = train_flat.mean(
    axis=0
)


feature_std = train_flat.std(
    axis=0
)


feature_std[
    feature_std < 1e-6
] = 1.0


def standardize(data):

    return (
        data - feature_mean
    ) / feature_std


X_train = standardize(
    X_train
)

X_val = standardize(
    X_val
)

X_test = standardize(
    X_test
)


print(
    "Training feature mean shape:",
    feature_mean.shape
)

print(
    "Training feature std shape:",
    feature_std.shape
)


# ============================================================
# SAVE SCALER
# ============================================================

scaler_data = {
    "mean": feature_mean,
    "std": feature_std
}


with open(
    SCALER_FILE,
    "wb"
) as file:

    pickle.dump(
        scaler_data,
        file
    )


print(
    "Scaler saved:",
    SCALER_FILE
)


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(

    torch.tensor(
        X_train,
        dtype=torch.float32
    ),

    torch.tensor(
        y_train,
        dtype=torch.float32
    )
)


val_dataset = TensorDataset(

    torch.tensor(
        X_val,
        dtype=torch.float32
    ),

    torch.tensor(
        y_val,
        dtype=torch.float32
    )
)


test_dataset = TensorDataset(

    torch.tensor(
        X_test,
        dtype=torch.float32
    ),

    torch.tensor(
        y_test,
        dtype=torch.float32
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TEMPORAL ATTENTION MODEL
# ============================================================

class TemporalAttentionModel(
    nn.Module
):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=128
    ):

        super().__init__()


        # --------------------------------------------------
        # Input normalization and dropout
        # --------------------------------------------------

        self.input_norm = nn.LayerNorm(input_dim)

        self.input_dropout = nn.Dropout(0.25)

        # --------------------------------------------------
        # Bidirectional LSTM
        # --------------------------------------------------

        self.lstm = nn.LSTM(

            input_size=input_dim,

            hidden_size=hidden_dim,

            num_layers=2,

            batch_first=True,

            dropout=0.2,

            bidirectional=True
        )


        attention_dim = (
            hidden_dim * 2
        )


        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                attention_dim,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )
        )


        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                attention_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(
        self,
        x
    ):

        # x:
        # batch × 8 × 512

        lstm_output, _ = self.lstm(
            x
        )

        # batch × 8 × 256

        attention_scores = (
            self.attention(
                lstm_output
            )
        )

        # batch × 8 × 1

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )


        # Weighted combination
        # of all frames

        context = torch.sum(

            attention_weights
            * lstm_output,

            dim=1
        )


        # batch × 256

        logits = self.classifier(
            context
        ).squeeze(1)


        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = TemporalAttentionModel(
    input_dim=FEATURE_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)


print("\nMODEL")
print("=====")

print(model)


# ============================================================
# CLASS WEIGHTING
# ============================================================

fake_count = np.sum(
    y_train == 0
)

real_count = np.sum(
    y_train == 1
)


positive_weight = (
    fake_count / real_count
)


print("\nCLASS WEIGHTING")
print("===============")

print(
    "Fake:",
    fake_count
)

print(
    "Real:",
    real_count
)

print(
    "Positive class weight:",
    round(
        positive_weight,
        4
    )
)


criterion = nn.BCEWithLogitsLoss(

    pos_weight=torch.tensor(
        positive_weight,
        dtype=torch.float32,
        device=device
    )
)


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    labels,
    threshold=0.50
):

    model.eval()

    probabilities = []


    with torch.no_grad():

        for batch_x, _ in loader:

            batch_x = batch_x.to(
                device
            )


            logits = model(
                batch_x
            )


            probs = torch.sigmoid(
                logits
            )


            probabilities.extend(
                probs.cpu().numpy()
            )


    probabilities = np.array(
        probabilities
    )


    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        labels,
        predictions
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions
        )
    )


    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    auc = roc_auc_score(
        labels,
        probabilities
    )


    return (
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        f1,
        auc,
        probabilities,
        predictions
    )


# ============================================================
# TRAINING
# ============================================================

print("\nTRAINING FINAL VIDEO MODEL")
print("==========================")

best_auc = -1.0

best_epoch = 0

best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0


    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            device
        )

        batch_y = batch_y.to(
            device
        )


        optimizer.zero_grad()


        logits = model(
            batch_x
        )


        loss = criterion(
            logits,
            batch_y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            * len(batch_x)
        )


    train_loss = (
        running_loss
        / len(train_dataset)
    )


    (
        val_acc,
        val_bal_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc,
        _,
        _
    ) = evaluate(
        model,
        val_loader,
        y_val,
        THRESHOLD
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"Loss={train_loss:.4f} "
        f"ValAcc={val_acc:.4f} "
        f"ValBalAcc={val_bal_acc:.4f} "
        f"ValF1={val_f1:.4f} "
        f"ValAUC={val_auc:.4f}"
    )


    # Save best model according to
    # validation ROC-AUC

    if val_auc > best_auc:

        best_auc = val_auc

        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)


print("\nBEST MODEL")
print("==========")

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation ROC-AUC:",
    round(
        best_auc,
        4
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_FILE
)


print(
    "\nModel saved:",
    MODEL_FILE
)


# ============================================================
# VALIDATION THRESHOLD CALIBRATION
# ============================================================

print("\nCALIBRATING VIDEO THRESHOLD")
print("===========================")

(
    _va,
    _vba,
    _vp,
    _vr,
    _vf,
    _vauc,
    calibration_probs,
    _
) = evaluate(
    model,
    val_loader,
    y_val,
    0.50
)

candidate_thresholds = np.arange(
    0.20,
    0.81,
    0.01
)

best_threshold = 0.50
best_balanced_accuracy = -1.0
best_threshold_f1 = -1.0

for candidate in candidate_thresholds:

    candidate_predictions = (
        calibration_probs >= candidate
    ).astype(int)

    candidate_balanced_accuracy = balanced_accuracy_score(
        y_val,
        candidate_predictions
    )

    candidate_f1 = f1_score(
        y_val,
        candidate_predictions,
        zero_division=0
    )

    if (
        candidate_balanced_accuracy > best_balanced_accuracy
        or (
            candidate_balanced_accuracy == best_balanced_accuracy
            and candidate_f1 > best_threshold_f1
        )
    ):

        best_balanced_accuracy = candidate_balanced_accuracy
        best_threshold_f1 = candidate_f1
        best_threshold = float(candidate)

THRESHOLD = best_threshold

print(
    "Calibrated threshold:",
    round(THRESHOLD, 4)
)

print(
    "Validation balanced accuracy:",
    round(best_balanced_accuracy, 4)
)

print(
    "Validation F1 at threshold:",
    round(best_threshold_f1, 4)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

(
    val_acc,
    val_bal_acc,
    val_precision,
    val_recall,
    val_f1,
    val_auc,
    val_probs,
    val_predictions
) = evaluate(
    model,
    val_loader,
    y_val,
    THRESHOLD
)


print("\nFINAL VALIDATION RESULTS")
print("========================")

print(
    "Accuracy          :",
    round(val_acc, 4)
)

print(
    "Balanced Accuracy :",
    round(val_bal_acc, 4)
)

print(
    "Precision         :",
    round(val_precision, 4)
)

print(
    "Recall            :",
    round(val_recall, 4)
)

print(
    "F1 Score          :",
    round(val_f1, 4)
)

print(
    "ROC-AUC           :",
    round(val_auc, 4)
)


print(
    "\nVALIDATION CONFUSION MATRIX"
)

print(
    confusion_matrix(
        y_val,
        val_predictions
    )
)


# ============================================================
# OFFICIAL TEST
# ============================================================

(
    test_acc,
    test_bal_acc,
    test_precision,
    test_recall,
    test_f1,
    test_auc,
    test_probs,
    test_predictions
) = evaluate(
    model,
    test_loader,
    y_test,
    THRESHOLD
)


print("\nOFFICIAL TEST RESULTS")
print("=====================")

print(
    "Accuracy          :",
    round(test_acc, 4)
)

print(
    "Balanced Accuracy :",
    round(test_bal_acc, 4)
)

print(
    "Precision         :",
    round(test_precision, 4)
)

print(
    "Recall            :",
    round(test_recall, 4)
)

print(
    "F1 Score          :",
    round(test_f1, 4)
)

print(
    "ROC-AUC           :",
    round(test_auc, 4)
)


print(
    "\nOFFICIAL TEST CONFUSION MATRIX"
)

test_cm = confusion_matrix(
    y_test,
    test_predictions
)

print(test_cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nOFFICIAL TEST CLASSIFICATION REPORT"
)

print(
    "===================================="
)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Fake",
            "Real"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "sequence_length":
        SEQUENCE_LENGTH,

    "feature_dim":
        FEATURE_DIM,

    "hidden_dim":
        HIDDEN_DIM,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "random_seed":
        RANDOM_SEED,

    "threshold":
        THRESHOLD,

    "threshold_method":
        "validation_balanced_accuracy",

    "best_validation_auc":
        float(best_auc),

    "architecture":
        "LayerNorm + Dropout + 2-layer Bidirectional LSTM + Self-Attention + Classifier",

    "model_type":
        "TemporalAttentionModel",

    "label_mapping": {
        0: "Fake",
        1: "Real"
    }
}


with open(
    CONFIG_FILE,
    "wb"
) as file:

    pickle.dump(
        config,
        file
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "best_epoch":
        best_epoch,

    "best_validation_auc":
        float(best_auc),

    "validation": {

        "accuracy":
            float(val_acc),

        "balanced_accuracy":
            float(val_bal_acc),

        "precision":
            float(val_precision),

        "recall":
            float(val_recall),

        "f1":
            float(val_f1),

        "roc_auc":
            float(val_auc),

        "confusion_matrix":
            confusion_matrix(
                y_val,
                val_predictions
            ).tolist()
    },

    "test": {

        "accuracy":
            float(test_acc),

        "balanced_accuracy":
            float(test_bal_acc),

        "precision":
            float(test_precision),

        "recall":
            float(test_recall),

        "f1":
            float(test_f1),

        "roc_auc":
            float(test_auc),

        "confusion_matrix":
            test_cm.tolist()
    }
}


with open(
    METRICS_FILE,
    "wb"
) as file:

    pickle.dump(
        metrics,
        file
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

correct = int(
    np.sum(
        test_predictions == y_test
    )
)

incorrect = (
    len(y_test) - correct
)


print("\nSAVED VIDEO MODEL ARTIFACTS")
print("===========================")

print(
    "1.",
    MODEL_FILE
)

print(
    "2.",
    SCALER_FILE
)

print(
    "3.",
    CONFIG_FILE
)

print(
    "4.",
    METRICS_FILE
)


print("\nFINAL TEST SUMMARY")
print("==================")

print(
    "Correct predictions:",
    correct
)

print(
    "Incorrect predictions:",
    incorrect
)

print(
    "Total test samples:",
    len(y_test)
)


print(
    "\nFINAL VIDEO MODEL TRAINING COMPLETED"
)