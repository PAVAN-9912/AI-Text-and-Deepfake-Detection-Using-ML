import random
import numpy as np
import pandas as pd

from pathlib import Path

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


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_PATH = Path(
    r"datasets\video\temporal_resnet_features.csv"
)

SEQUENCE_LENGTH = 8
FEATURE_DIMENSION = 512

HIDDEN_SIZE = 128
NUM_LAYERS = 2

BATCH_SIZE = 64
EPOCHS = 15

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# HEADER
# ============================================================

print(
    "TEMPORAL VIDEO LSTM MODEL"
)

print(
    "========================="
)

print(
    "Device:",
    DEVICE
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Feature dimension:",
    FEATURE_DIMENSION
)


# ============================================================
# LOAD TEMPORAL FEATURES
# ============================================================

if not FEATURE_PATH.exists():

    raise FileNotFoundError(
        f"Temporal feature file not found:\n"
        f"{FEATURE_PATH}"
    )


features = pd.read_csv(
    FEATURE_PATH
)


print(
    "\nTotal samples:",
    len(features)
)


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in features.columns
    if column.startswith("frame_")
    and "_feature_" in column
]


expected_feature_count = (
    SEQUENCE_LENGTH
    *
    FEATURE_DIMENSION
)


print(
    "Temporal feature columns:",
    len(feature_columns)
)

print(
    "Expected:",
    expected_feature_count
)


if len(feature_columns) != expected_feature_count:

    raise ValueError(
        "Incorrect number of temporal features.\n"
        f"Expected: {expected_feature_count}\n"
        f"Found: {len(feature_columns)}"
    )


# ============================================================
# SORT FEATURE COLUMNS CORRECTLY
# ============================================================

def feature_sort_key(column):

    # Example:
    #
    # frame_1_feature_0
    # frame_1_feature_1
    # ...
    # frame_8_feature_511

    parts = column.split("_")

    frame_number = int(
        parts[1]
    )

    feature_number = int(
        parts[3]
    )

    return (
        frame_number,
        feature_number
    )


feature_columns = sorted(
    feature_columns,
    key=feature_sort_key
)


# ============================================================
# VERIFY METADATA
# ============================================================

required_columns = [
    "split",
    "label",
    "label_name",
    "video_id",
    "filename"
]


missing_columns = [
    column
    for column in required_columns
    if column not in features.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        +
        str(missing_columns)
    )


# ============================================================
# VERIFY VIDEO IDs
# ============================================================

if features[
    "video_id"
].duplicated().any():

    duplicate_count = (
        features["video_id"]
        .duplicated()
        .sum()
    )

    raise ValueError(
        f"Duplicate video IDs found: "
        f"{duplicate_count}"
    )


print(
    "Unique video IDs:",
    features["video_id"].nunique()
)


# ============================================================
# CREATE TEMPORAL FEATURE MATRIX
# ============================================================

X_all = features[
    feature_columns
].values.astype(
    np.float32
)


print(
    "\nFlat feature matrix:",
    X_all.shape
)


# Expected:

# (6529, 4096)


if X_all.shape[1] != expected_feature_count:

    raise ValueError(
        "Unexpected feature matrix shape."
    )


# ============================================================
# RESHAPE INTO SEQUENCES
# ============================================================

X_all = X_all.reshape(
    -1,
    SEQUENCE_LENGTH,
    FEATURE_DIMENSION
)


print(
    "Temporal feature matrix:",
    X_all.shape
)


# Expected:

# (6529, 8, 512)


# ============================================================
# LABELS
# ============================================================

y_all = features[
    "label"
].values.astype(
    np.int64
)


# ============================================================
# SPLIT MASKS
# ============================================================

train_mask = (
    features["split"]
    ==
    "train"
).values


validation_mask = (
    features["split"]
    ==
    "validation"
).values


test_mask = (
    features["split"]
    ==
    "test"
).values


# ============================================================
# CREATE SPLITS
# ============================================================

X_train = X_all[
    train_mask
]

y_train = y_all[
    train_mask
]


X_validation = X_all[
    validation_mask
]

y_validation = y_all[
    validation_mask
]


X_test = X_all[
    test_mask
]

y_test = y_all[
    test_mask
]


print(
    "\nDATASET SPLIT"
)

print(
    "=============="
)

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_validation.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nTRAIN LABEL DISTRIBUTION"
)

print(
    pd.Series(
        y_train
    ).value_counts()
    .sort_index()
)


print(
    "\nVALIDATION LABEL DISTRIBUTION"
)

print(
    pd.Series(
        y_validation
    ).value_counts()
    .sort_index()
)


print(
    "\nTEST LABEL DISTRIBUTION"
)

print(
    pd.Series(
        y_test
    ).value_counts()
    .sort_index()
)


# ============================================================
# DATASET
# ============================================================

class VideoDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )


    def __len__(
        self
    ):

        return len(
            self.X
        )


    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = VideoDataset(
    X_train,
    y_train
)


validation_dataset = VideoDataset(
    X_validation,
    y_validation
)


test_dataset = VideoDataset(
    X_test,
    y_test
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LSTM MODEL
# ============================================================

class TemporalLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers
    ):

        super().__init__()


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
            if num_layers > 1
            else 0
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
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

        output, _ = self.lstm(
            x
        )


        # Last temporal state

        last_output = (
            output[:, -1, :]
        )


        logits = self.classifier(
            last_output
        )


        return logits.squeeze(1)


# ============================================================
# CREATE MODEL
# ============================================================

model = TemporalLSTM(
    input_size=FEATURE_DIMENSION,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(
    DEVICE
)


print(
    "\nMODEL"
)

print(
    "====="
)

print(
    model
)


# ============================================================
# CLASS WEIGHTING
# ============================================================

fake_count = np.sum(
    y_train == 0
)

real_count = np.sum(
    y_train == 1
)


# Positive class = Real

pos_weight = (
    fake_count /
    real_count
)


print(
    "\nCLASS WEIGHTING"
)

print(
    "==============="
)

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
        pos_weight,
        4
    )
)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=DEVICE
    )
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()


    all_probabilities = []
    all_labels = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )


            logits = model(
                X_batch
            )


            probabilities = (
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )


            all_probabilities.extend(
                probabilities
            )


            all_labels.extend(
                y_batch.numpy()
            )


    return (
        np.asarray(
            all_labels
        ).astype(int),

        np.asarray(
            all_probabilities
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    metrics = {}


    metrics["accuracy"] = (
        accuracy_score(
            y_true,
            predictions
        )
    )


    metrics["balanced_accuracy"] = (
        balanced_accuracy_score(
            y_true,
            predictions
        )
    )


    metrics["precision"] = (
        precision_score(
            y_true,
            predictions,
            zero_division=0
        )
    )


    metrics["recall"] = (
        recall_score(
            y_true,
            predictions,
            zero_division=0
        )
    )


    metrics["f1"] = (
        f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    )


    try:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities
            )
        )

    except ValueError:

        metrics["roc_auc"] = 0.0


    return metrics


# ============================================================
# TRAIN
# ============================================================

print(
    "\nTRAINING TEMPORAL LSTM"
)

print(
    "======================"
)


best_f1 = -1.0

best_state = None

best_epoch = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()


    total_loss = 0.0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item()
            *
            len(X_batch)
        )


    average_loss = (
        total_loss
        /
        len(train_dataset)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    y_val_epoch, p_val_epoch = (
        evaluate(
            model,
            validation_loader
        )
    )


    val_metrics = calculate_metrics(
        y_val_epoch,
        p_val_epoch
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"Loss={average_loss:.4f} "
        f"ValAcc={val_metrics['accuracy']:.4f} "
        f"ValBalAcc={val_metrics['balanced_accuracy']:.4f} "
        f"ValF1={val_metrics['f1']:.4f} "
        f"ValAUC={val_metrics['roc_auc']:.4f}"
    )


    # --------------------------------------------------------
    # Save best model based on validation F1
    # --------------------------------------------------------

    if (
        val_metrics["f1"]
        >
        best_f1
    ):

        best_f1 = (
            val_metrics["f1"]
        )

        best_epoch = epoch


        best_state = {
            key: value.detach()
            .cpu()
            .clone()

            for key, value
            in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is None:

    raise RuntimeError(
        "No best model was saved."
    )


model.load_state_dict(
    best_state
)


print(
    "\nBEST MODEL"
)

print(
    "==========="
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation F1:",
    round(
        best_f1,
        4
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

y_val, p_val = evaluate(
    model,
    validation_loader
)


val_metrics = calculate_metrics(
    y_val,
    p_val
)


val_predictions = (
    p_val >= 0.5
).astype(int)


print(
    "\nFINAL VALIDATION RESULTS"
)

print(
    "========================="
)

print(
    "Accuracy          :",
    round(
        val_metrics["accuracy"],
        4
    )
)

print(
    "Balanced Accuracy :",
    round(
        val_metrics["balanced_accuracy"],
        4
    )
)

print(
    "Precision         :",
    round(
        val_metrics["precision"],
        4
    )
)

print(
    "Recall            :",
    round(
        val_metrics["recall"],
        4
    )
)

print(
    "F1 Score          :",
    round(
        val_metrics["f1"],
        4
    )
)

print(
    "ROC-AUC           :",
    round(
        val_metrics["roc_auc"],
        4
    )
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

y_test_result, p_test = evaluate(
    model,
    test_loader
)


test_metrics = calculate_metrics(
    y_test_result,
    p_test
)


test_predictions = (
    p_test >= 0.5
).astype(int)


print(
    "\nOFFICIAL TEST RESULTS"
)

print(
    "====================="
)

print(
    "Accuracy          :",
    round(
        test_metrics["accuracy"],
        4
    )
)

print(
    "Balanced Accuracy :",
    round(
        test_metrics["balanced_accuracy"],
        4
    )
)

print(
    "Precision         :",
    round(
        test_metrics["precision"],
        4
    )
)

print(
    "Recall            :",
    round(
        test_metrics["recall"],
        4
    )
)

print(
    "F1 Score          :",
    round(
        test_metrics["f1"],
        4
    )
)

print(
    "ROC-AUC           :",
    round(
        test_metrics["roc_auc"],
        4
    )
)


print(
    "\nOFFICIAL TEST CONFUSION MATRIX"
)

print(
    "==============================="
)

print(
    confusion_matrix(
        y_test_result,
        test_predictions
    )
)


print(
    "\nOFFICIAL TEST CLASSIFICATION REPORT"
)

print(
    "===================================="
)

print(
    classification_report(
        y_test_result,
        test_predictions,
        target_names=[
            "Fake",
            "Real"
        ],
        zero_division=0
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

correct = np.sum(
    test_predictions
    ==
    y_test_result
)


incorrect = (
    len(y_test_result)
    -
    correct
)


print(
    "\nFINAL TEST SUMMARY"
)

print(
    "=================="
)

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
    len(
        y_test_result
    )
)


print(
    "\nTEMPORAL LSTM MODEL COMPLETED"
)