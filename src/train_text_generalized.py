import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
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

TRAIN_FILE = (
    r"datasets\text\HC3\train.csv"
)

VALIDATION_FILE = (
    r"datasets\text\HC3\validation.csv"
)

MODEL_DIR = (
    r"models\text_generalized"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("GENERALIZED TEXT DETECTOR")
print("=========================")

print(
    "Using full HC3 training dataset"
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


print(
    "Training samples:",
    len(train_df)
)


print(
    "Validation samples:",
    len(validation_df)
)


# ============================================================
# VERIFY DATA
# ============================================================

required_columns = [
    "text",
    "label"
]


for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Training dataset missing: {column}"
        )


    if column not in validation_df.columns:

        raise ValueError(
            f"Validation dataset missing: {column}"
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


y_train = (
    train_df["label"]
    .values
)


y_validation = (
    validation_df["label"]
    .values
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

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
# WORD TF-IDF
# ============================================================

print()
print("BUILDING WORD TF-IDF")
print("====================")


word_vectorizer = TfidfVectorizer(

    analyzer="word",

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95,

    max_features=150000,

    sublinear_tf=True,

    strip_accents="unicode",

    lowercase=True
)


print(
    "Fitting word vocabulary..."
)


X_train_word = (
    word_vectorizer.fit_transform(
        train_df["text"]
    )
)


print(
    "Transforming validation..."
)


X_validation_word = (
    word_vectorizer.transform(
        validation_df["text"]
    )
)


print(
    "Training word shape:",
    X_train_word.shape
)


print(
    "Validation word shape:",
    X_validation_word.shape
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print()
print("BUILDING CHARACTER TF-IDF")
print("=========================")


char_vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(3, 5),

    min_df=3,

    max_features=150000,

    sublinear_tf=True,

    lowercase=True
)


print(
    "Fitting character vocabulary..."
)


X_train_char = (
    char_vectorizer.fit_transform(
        train_df["text"]
    )
)


print(
    "Transforming validation..."
)


X_validation_char = (
    char_vectorizer.transform(
        validation_df["text"]
    )
)


print(
    "Training character shape:",
    X_train_char.shape
)


print(
    "Validation character shape:",
    X_validation_char.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


X_train = hstack(
    [
        X_train_word,
        X_train_char
    ]
).tocsr()


X_validation = hstack(
    [
        X_validation_word,
        X_validation_char
    ]
).tocsr()


print(
    "Combined training shape:",
    X_train.shape
)


print(
    "Combined validation shape:",
    X_validation.shape
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("TRAINING LOGISTIC REGRESSION")
print("============================")


classifier = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    C=2.0,

    solver="liblinear",

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


validation_predictions = (
    classifier.predict(
        X_validation
    )
)


validation_probabilities = (
    classifier.predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


auc = roc_auc_score(
    y_validation,
    validation_probabilities
)


# ============================================================
# RESULTS
# ============================================================

print()
print("GENERALIZED TEXT MODEL RESULTS")
print("===============================")


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
    validation_predictions
)


print(
    cm
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
        validation_predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

print()
print("SAVING MODEL ARTIFACTS")
print("======================")


joblib.dump(
    word_vectorizer,
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)


joblib.dump(
    char_vectorizer,
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)


joblib.dump(
    classifier,
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


config = {

    "model_type":
        "word_char_tfidf_logistic_regression",

    "word_ngram_range":
        (1, 2),

    "char_ngram_range":
        (3, 5),

    "word_max_features":
        150000,

    "char_max_features":
        150000,

    "class_weight":
        "balanced",

    "C":
        2.0,

    "training_samples":
        len(train_df),

    "validation_samples":
        len(validation_df),

    "feature_count":
        int(X_train.shape[1])
}


joblib.dump(
    config,
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
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
    " - word_tfidf_vectorizer.pkl"
)

print(
    " - char_tfidf_vectorizer.pkl"
)

print(
    " - text_classifier.pkl"
)

print(
    " - text_config.pkl"
)

print(
    " - text_metrics.pkl"
)


print()
print(
    "GENERALIZED TEXT MODEL TRAINING COMPLETED"
)