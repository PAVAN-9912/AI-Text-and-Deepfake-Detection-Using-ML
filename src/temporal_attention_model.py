import copy
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

SEQUENCE_LENGTH = 8
FEATURE_DIM = 512

HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("TEMPORAL ATTENTION VIDEO MODEL")
print("==============================")
print("Device:", device)
print("Sequence length:", SEQUENCE_LENGTH)
print("Feature dimension:", FEATURE_DIM)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("\nDATASET")
print("=======")
print("Total samples:", len(df))


# ============================================================
# FIND FEATURE COLUMNS
# ============================================================

# These columns are metadata and must NOT be treated
# as ResNet feature columns.

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
    c for c in df.columns
    if c not in metadata_columns
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
# CHECK VIDEO IDS
# ============================================================

print(
    "Unique video IDs:",
    df["video_id"].nunique()
)

assert (
    df["video_id"].nunique()
    == len(df)
), "Duplicate video IDs detected!"


# ============================================================
# CREATE TEMPORAL MATRIX
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
    df["split"].values
    == "train"
)

val_mask = (
    df["split"].values
    == "validation"
)

test_mask = (
    df["split"].values
    == "test"
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

print(
    "\nTRAIN LABEL DISTRIBUTION"
)

print(
    pd.Series(
        y_train
    ).value_counts().sort_index()
)


print(
    "\nVALIDATION LABEL DISTRIBUTION"
)

print(
    pd.Series(
        y_val
    ).value_counts().sort_index()
)


print(
    "\nTEST LABEL DISTRIBUTION"
)

print(
    pd.Series(
        y_test
    ).value_counts().sort_index()
)


# ============================================================
# STANDARDIZATION
# ============================================================
# Fit standardization ONLY on training data.

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


        # Bidirectional LSTM processes
        # the 8 video frames as a sequence.

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


        # Attention learns which frames
        # are more important.

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


        # Final classifier.

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


    def forward(self, x):

        # Input:
        # batch × 8 × 512

        lstm_output, _ = self.lstm(
            x
        )

        # Output:
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


        # Weighted combination of
        # all 8 frames.

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
    fake_count
    / real_count
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
    labels
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
        probabilities >= 0.50
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

print(
    "\nTRAINING TEMPORAL ATTENTION MODEL"
)

print(
    "================================="
)


best_f1 = -1

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
        y_val
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"Loss={train_loss:.4f} "
        f"ValAcc={val_acc:.4f} "
        f"ValBalAcc={val_bal_acc:.4f} "
        f"ValF1={val_f1:.4f} "
        f"ValAUC={val_auc:.4f}"
    )


    if val_f1 > best_f1:

        best_f1 = val_f1

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
    "Best validation F1:",
    round(
        best_f1,
        4
    )
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
    y_val
)


print(
    "\nFINAL VALIDATION RESULTS"
)

print(
    "========================="
)


print(
    "Accuracy          :",
    round(
        val_acc,
        4
    )
)


print(
    "Balanced Accuracy :",
    round(
        val_bal_acc,
        4
    )
)


print(
    "Precision         :",
    round(
        val_precision,
        4
    )
)


print(
    "Recall            :",
    round(
        val_recall,
        4
    )
)


print(
    "F1 Score          :",
    round(
        val_f1,
        4
    )
)


print(
    "ROC-AUC           :",
    round(
        val_auc,
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
    y_test
)


print(
    "\nOFFICIAL TEST RESULTS"
)

print(
    "====================="
)


print(
    "Accuracy          :",
    round(
        test_acc,
        4
    )
)


print(
    "Balanced Accuracy :",
    round(
        test_bal_acc,
        4
    )
)


print(
    "Precision         :",
    round(
        test_precision,
        4
    )
)


print(
    "Recall            :",
    round(
        test_recall,
        4
    )
)


print(
    "F1 Score          :",
    round(
        test_f1,
        4
    )
)


print(
    "ROC-AUC           :",
    round(
        test_auc,
        4
    )
)


# ============================================================
# TEST CONFUSION MATRIX
# ============================================================

print(
    "\nOFFICIAL TEST CONFUSION MATRIX"
)

print(
    "==============================="
)


print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


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
# TEST SUMMARY
# ============================================================

correct = np.sum(
    test_predictions == y_test
)


incorrect = (
    len(y_test)
    - correct
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
    len(y_test)
)


print(
    "\nTEMPORAL ATTENTION MODEL COMPLETED"
)