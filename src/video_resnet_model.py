import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = (
    r"datasets\video\resnet_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    FEATURES_PATH
)


print(
    "VIDEO RESNET-18 BASELINE"
)

print(
    "========================"
)

print(
    "Total samples:",
    len(df)
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith(
        "feature_"
    )
]


print(
    "Feature dimensions:",
    len(feature_columns)
)


# ============================================================
# SPLIT DATA
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()


validation_df = df[
    df["split"] == "validation"
].copy()


test_df = df[
    df["split"] == "test"
].copy()


print(
    "\nTRAINING SAMPLES:",
    len(train_df)
)

print(
    "VALIDATION SAMPLES:",
    len(validation_df)
)

print(
    "TEST SAMPLES:",
    len(test_df)
)


# ============================================================
# LABELS
# ============================================================

y_train = train_df[
    "label"
].values


y_validation = validation_df[
    "label"
].values


y_test = test_df[
    "label"
].values


# ============================================================
# FEATURES
# ============================================================

X_train = train_df[
    feature_columns
].values


X_validation = validation_df[
    feature_columns
].values


X_test = test_df[
    feature_columns
].values


# ============================================================
# CHECK FEATURES
# ============================================================

print(
    "\nFEATURE MATRICES"
)

print(
    "Train:",
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
# STANDARDIZATION
# ============================================================

print(
    "\nStandardizing features..."
)


scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_validation = scaler.transform(
    X_validation
)


X_test = scaler.transform(
    X_test
)


# ============================================================
# CLASSIFIER
# ============================================================

print(
    "\nTraining Logistic Regression..."
)


classifier = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)


classifier.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nEvaluating validation set..."
)


validation_predictions = (
    classifier.predict(
        X_validation
    )
)


# ============================================================
# TEST
# ============================================================

print(
    "Evaluating official test set..."
)


test_predictions = (
    classifier.predict(
        X_test
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def print_metrics(
    name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    print(
        f"\n{name}"
    )

    print(
        "=" * len(name)
    )


    print(
        "Accuracy :",
        round(
            accuracy,
            4
        )
    )


    print(
        "Precision:",
        round(
            precision,
            4
        )
    )


    print(
        "Recall   :",
        round(
            recall,
            4
        )
    )


    print(
        "F1 Score :",
        round(
            f1,
            4
        )
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )


# ============================================================
# RESULTS
# ============================================================

print_metrics(
    "VALIDATION RESULTS",
    y_validation,
    validation_predictions
)


print_metrics(
    "OFFICIAL TEST RESULTS",
    y_test,
    test_predictions
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
# PREDICTION COUNTS
# ============================================================

print(
    "\nTEST PREDICTION DISTRIBUTION"
)

print(
    pd.Series(
        test_predictions
    ).value_counts().sort_index()
)


# ============================================================
# CORRECT / INCORRECT
# ============================================================

correct = np.sum(
    test_predictions
    == y_test
)


incorrect = np.sum(
    test_predictions
    != y_test
)


print(
    "\nTEST SUMMARY"
)

print(
    "============"
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