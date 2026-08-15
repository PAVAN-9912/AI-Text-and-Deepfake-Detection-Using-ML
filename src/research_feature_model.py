import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = r"datasets\text\HC3\research_features.csv"


# --------------------------------------------------
# Load research features
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("RESEARCH FEATURE MODEL")
print("======================")

print("Total samples:", len(df))


# --------------------------------------------------
# Create missing-value indicator
# --------------------------------------------------

df["burstiness_missing"] = (
    df["burstiness"].isna().astype(int)
)


# --------------------------------------------------
# Log-transform skewed features
# --------------------------------------------------

df["log_perplexity"] = np.log1p(
    df["perplexity"]
)

df["log_burstiness"] = np.log1p(
    df["burstiness"]
)


# --------------------------------------------------
# Select features
# --------------------------------------------------

feature_columns = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity",
    "log_perplexity",
    "log_burstiness",
    "burstiness_missing"
]


X = df[feature_columns]
y = df["label"]


print("\nFeatures:")
print(feature_columns)


# --------------------------------------------------
# Model pipeline
# --------------------------------------------------

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
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
# 5-fold stratified cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


accuracies = []
precisions = []
recalls = []
f1_scores = []


print("\n5-FOLD CROSS-VALIDATION")
print("=======================")


for fold, (train_index, val_index) in enumerate(
    cv.split(X, y),
    start=1
):

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    y_train = y.iloc[train_index]
    y_val = y.iloc[val_index]


    # Fit ONLY on the fold's training portion
    pipeline.fit(X_train, y_train)


    predictions = pipeline.predict(X_val)


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


    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


    print(
        f"Fold {fold}: "
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )


# --------------------------------------------------
# Average results
# --------------------------------------------------

print("\nAVERAGE RESULTS")
print("===============")

print(
    "Accuracy :",
    round(np.mean(accuracies), 4)
)

print(
    "Precision:",
    round(np.mean(precisions), 4)
)

print(
    "Recall   :",
    round(np.mean(recalls), 4)
)

print(
    "F1 Score :",
    round(np.mean(f1_scores), 4)
)


# --------------------------------------------------
# Standard deviation
# --------------------------------------------------

print("\nSTANDARD DEVIATION")
print("==================")

print(
    "Accuracy :",
    round(np.std(accuracies), 4)
)

print(
    "Precision:",
    round(np.std(precisions), 4)
)

print(
    "Recall   :",
    round(np.std(recalls), 4)
)

print(
    "F1 Score :",
    round(np.std(f1_scores), 4)
)