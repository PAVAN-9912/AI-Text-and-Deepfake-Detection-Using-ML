import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_FEATURES = (
    r"datasets\text\HC3\research_features.csv"
)

VALIDATION_FEATURES = (
    r"datasets\text\HC3\validation_research_features.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_FEATURES)
val_df = pd.read_csv(VALIDATION_FEATURES)


print("RESEARCH FEATURE VALIDATION")
print("===========================")

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# --------------------------------------------------
# Create transformed features
# --------------------------------------------------

for df in [train_df, val_df]:

    df["burstiness_missing"] = (
        df["burstiness"].isna().astype(int)
    )

    df["log_perplexity"] = np.log1p(
        df["perplexity"]
    )

    df["log_burstiness"] = np.log1p(
        df["burstiness"]
    )


# --------------------------------------------------
# Frozen feature set
# --------------------------------------------------

features = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity",
    "log_perplexity",
    "log_burstiness",
    "burstiness_missing"
]


X_train = train_df[features]
y_train = train_df["label"]

X_val = val_df[features]
y_val = val_df["label"]


# --------------------------------------------------
# Model
# --------------------------------------------------

pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# --------------------------------------------------
# Train ONLY on training-derived data
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Validation prediction
# --------------------------------------------------

print("Evaluating on unseen validation data...")

predictions = pipeline.predict(
    X_val
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_val,
    predictions
)

precision = precision_score(
    y_val,
    predictions
)

recall = recall_score(
    y_val,
    predictions
)

f1 = f1_score(
    y_val,
    predictions
)


print("\nVALIDATION RESULTS")
print("==================")

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_val,
    predictions
)


print("\nCONFUSION MATRIX")
print("================")

print(cm)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "Human",
            "AI"
        ]
    )
)