import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score
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
    "BALANCED VIDEO RESNET-18 MODEL"
)

print(
    "=============================="
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
    if column.startswith("feature_")
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
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nTRAIN LABEL DISTRIBUTION"
)

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nVALIDATION LABEL DISTRIBUTION"
)

print(
    validation_df["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nTEST LABEL DISTRIBUTION"
)

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FEATURES AND LABELS
# ============================================================

X_train = train_df[
    feature_columns
].values

y_train = train_df[
    "label"
].values


X_validation = validation_df[
    feature_columns
].values

y_validation = validation_df[
    "label"
].values


X_test = test_df[
    feature_columns
].values

y_test = test_df[
    "label"
].values


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
# BALANCED LOGISTIC REGRESSION
# ============================================================

print(
    "\nTraining balanced Logistic Regression..."
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
# PREDICTIONS
# ============================================================

print(
    "Generating predictions..."
)


validation_probabilities = (
    classifier.predict_proba(
        X_validation
    )[:, 1]
)


test_probabilities = (
    classifier.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

validation_predictions = (
    validation_probabilities >= 0.50
).astype(int)


test_predictions = (
    test_probabilities >= 0.50
).astype(int)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    predictions
):

    return {
        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
    }


# ============================================================
# VALIDATION RESULTS
# ============================================================

validation_metrics = calculate_metrics(
    y_validation,
    validation_predictions
)


print(
    "\nVALIDATION RESULTS"
)

print(
    "=================="
)

print(
    "Accuracy          :",
    round(
        validation_metrics["accuracy"],
        4
    )
)

print(
    "Balanced Accuracy :",
    round(
        validation_metrics["balanced_accuracy"],
        4
    )
)

print(
    "Precision         :",
    round(
        validation_metrics["precision"],
        4
    )
)

print(
    "Recall            :",
    round(
        validation_metrics["recall"],
        4
    )
)

print(
    "F1 Score          :",
    round(
        validation_metrics["f1"],
        4
    )
)


print(
    "\nValidation Confusion Matrix:"
)

print(
    confusion_matrix(
        y_validation,
        validation_predictions
    )
)


# ============================================================
# VALIDATION ROC-AUC
# ============================================================

validation_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)


print(
    "Validation ROC-AUC:",
    round(
        validation_auc,
        4
    )
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print(
    "\nTHRESHOLD ANALYSIS"
)

print(
    "=================="
)


threshold_results = []


thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


for threshold in thresholds:

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


    metrics = calculate_metrics(
        y_validation,
        predictions
    )


    threshold_results.append(
        {
            "threshold":
                threshold,

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics["balanced_accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"]
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print(
    threshold_df.round(4).to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df[
        "f1"
    ].idxmax()
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
    round(
        best_threshold,
        2
    )
)

print(
    "Validation F1:",
    round(
        best_row["f1"],
        4
    )
)


# ============================================================
# TEST WITH DEFAULT THRESHOLD
# ============================================================

test_metrics_default = calculate_metrics(
    y_test,
    test_predictions
)


print(
    "\nOFFICIAL TEST RESULTS"
)

print(
    "====================="
)

print(
    "Threshold         : 0.50"
)

print(
    "Accuracy          :",
    round(
        test_metrics_default["accuracy"],
        4
    )
)

print(
    "Balanced Accuracy :",
    round(
        test_metrics_default["balanced_accuracy"],
        4
    )
)

print(
    "Precision         :",
    round(
        test_metrics_default["precision"],
        4
    )
)

print(
    "Recall            :",
    round(
        test_metrics_default["recall"],
        4
    )
)

print(
    "F1 Score          :",
    round(
        test_metrics_default["f1"],
        4
    )
)


print(
    "\nTest Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# TEST WITH VALIDATION-SELECTED THRESHOLD
# ============================================================

test_predictions_tuned = (
    test_probabilities
    >= best_threshold
).astype(int)


test_metrics_tuned = calculate_metrics(
    y_test,
    test_predictions_tuned
)


print(
    "\nOFFICIAL TEST RESULTS - TUNED THRESHOLD"
)

print(
    "========================================"
)

print(
    "Threshold         :",
    round(
        best_threshold,
        2
    )
)

print(
    "Accuracy          :",
    round(
        test_metrics_tuned["accuracy"],
        4
    )
)

print(
    "Balanced Accuracy :",
    round(
        test_metrics_tuned["balanced_accuracy"],
        4
    )
)

print(
    "Precision         :",
    round(
        test_metrics_tuned["precision"],
        4
    )
)

print(
    "Recall            :",
    round(
        test_metrics_tuned["recall"],
        4
    )
)

print(
    "F1 Score          :",
    round(
        test_metrics_tuned["f1"],
        4
    )
)


print(
    "\nTuned Test Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        test_predictions_tuned
    )
)


# ============================================================
# ROC-AUC
# ============================================================

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


print(
    "\nOFFICIAL TEST ROC-AUC:",
    round(
        test_auc,
        4
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
        test_predictions_tuned,
        target_names=[
            "Fake",
            "Real"
        ],
        zero_division=0
    )
)


# ============================================================
# SUMMARY
# ============================================================

correct = np.sum(
    test_predictions_tuned
    == y_test
)

incorrect = np.sum(
    test_predictions_tuned
    != y_test
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