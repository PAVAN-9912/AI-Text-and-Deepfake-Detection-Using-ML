import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

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

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


print("TEMPORAL LSTM THRESHOLD ANALYSIS")
print("================================")
print("Device:", DEVICE)


# ============================================================
# LOAD FEATURES
# ============================================================

df = pd.read_csv(
    FEATURE_PATH
)


feature_columns = [
    column
    for column in df.columns
    if column.startswith("frame_")
    and "_feature_" in column
]


def feature_sort_key(column):

    parts = column.split("_")

    return (
        int(parts[1]),
        int(parts[3])
    )


feature_columns = sorted(
    feature_columns,
    key=feature_sort_key
)


X = df[
    feature_columns
].values.astype(
    np.float32
)


X = X.reshape(
    -1,
    SEQUENCE_LENGTH,
    FEATURE_DIMENSION
)


y = df[
    "label"
].values.astype(
    np.int64
)


# ============================================================
# SPLITS
# ============================================================

train_mask = (
    df["split"] == "train"
).values

validation_mask = (
    df["split"] == "validation"
).values

test_mask = (
    df["split"] == "test"
).values


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[validation_mask]
y_val = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nDATASET")
print("=======")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# DATASET
# ============================================================

class VideoDataset(
    torch.utils.data.Dataset
):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


train_loader = torch.utils.data.DataLoader(
    VideoDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    VideoDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    VideoDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class TemporalLSTM(nn.Module):

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
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.classifier(
            last_output
        ).squeeze(1)


model = TemporalLSTM(
    FEATURE_DIMENSION,
    HIDDEN_SIZE,
    NUM_LAYERS
).to(DEVICE)


# ============================================================
# CLASS WEIGHT
# ============================================================

fake_count = np.sum(
    y_train == 0
)

real_count = np.sum(
    y_train == 1
)

pos_weight = (
    fake_count / real_count
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
# EVALUATION FUNCTION
# ============================================================

def get_predictions(loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            logits = model(
                X_batch
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                y_batch.numpy()
            )

    return (
        np.asarray(labels).astype(int),
        np.asarray(probabilities)
    )


# ============================================================
# TRAIN
# ============================================================

print("\nTRAINING")
print("========")

best_f1 = -1
best_state = None
best_epoch = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0

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
        total_loss /
        len(X_train)
    )

    y_epoch, p_epoch = get_predictions(
        val_loader
    )

    predictions = (
        p_epoch >= 0.5
    ).astype(int)

    f1 = f1_score(
        y_epoch,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_epoch,
        p_epoch
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"Loss={average_loss:.4f} "
        f"ValF1={f1:.4f} "
        f"ValAUC={auc:.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_epoch = epoch

        best_state = {
            k: v.detach()
            .cpu()
            .clone()

            for k, v
            in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)


print("\nBEST MODEL")
print("===========")

print("Best epoch:", best_epoch)
print("Best validation F1:", round(best_f1, 4))


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

y_val, p_val = get_predictions(
    val_loader
)


# ============================================================
# VALIDATION THRESHOLD ANALYSIS
# ============================================================

print("\nVALIDATION THRESHOLD ANALYSIS")
print("=============================")

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


results = []


for threshold in thresholds:

    predictions = (
        p_val >= threshold
    ).astype(int)

    results.append({

        "threshold": round(
            threshold,
            2
        ),

        "accuracy": accuracy_score(
            y_val,
            predictions
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_val,
                predictions
            ),

        "precision":
            precision_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_val,
                predictions,
                zero_division=0
            )
    })


threshold_df = pd.DataFrame(
    results
)


print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# SELECT BEST F1 THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


best_threshold = float(
    best_row["threshold"]
)


print(
    "\nBEST VALIDATION THRESHOLD"
)

print(
    "========================="
)

print(
    "Threshold:",
    best_threshold
)

print(
    "F1:",
    round(
        best_row["f1"],
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        best_row["balanced_accuracy"],
        4
    )
)


# ============================================================
# OFFICIAL TEST
# ============================================================

y_test, p_test = get_predictions(
    test_loader
)


# ============================================================
# TEST AT 0.50
# ============================================================

test_default = (
    p_test >= 0.50
).astype(int)


print(
    "\nOFFICIAL TEST — THRESHOLD 0.50"
)

print(
    "=============================="
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_default
        ),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test,
            test_default
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_default,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_default,
            zero_division=0
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            test_default,
            zero_division=0
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            p_test
        ),
        4
    )
)


# ============================================================
# TEST USING VALIDATION-SELECTED THRESHOLD
# ============================================================

test_tuned = (
    p_test >= best_threshold
).astype(int)


print(
    "\nOFFICIAL TEST — VALIDATION TUNED THRESHOLD"
)

print(
    "=========================================="
)

print(
    "Threshold:",
    best_threshold
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_tuned
        ),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test,
            test_tuned
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_tuned,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_tuned,
            zero_division=0
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            test_tuned,
            zero_division=0
        ),
        4
    )
)


print(
    "\nTUNED TEST CONFUSION MATRIX"
)

print(
    confusion_matrix(
        y_test,
        test_tuned
    )
)


print(
    "\nTUNED TEST CLASSIFICATION REPORT"
)

print(
    "================================="
)

print(
    classification_report(
        y_test,
        test_tuned,
        target_names=[
            "Fake",
            "Real"
        ],
        zero_division=0
    )
)


print(
    "\nTEMPORAL VIDEO THRESHOLD ANALYSIS COMPLETED"
)