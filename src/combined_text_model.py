import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


# ==================================================
# FILE PATHS
# ==================================================

TRAIN_TEXT = r"datasets\text\HC3\feature_development.csv"
TRAIN_FEATURES = r"datasets\text\HC3\research_features.csv"

VAL_TEXT = r"datasets\text\HC3\feature_validation.csv"
VAL_FEATURES = r"datasets\text\HC3\validation_research_features.csv"


# ==================================================
# LOAD DATA
# ==================================================

train_text = pd.read_csv(TRAIN_TEXT)
train_features = pd.read_csv(TRAIN_FEATURES)

val_text = pd.read_csv(VAL_TEXT)
val_features = pd.read_csv(VAL_FEATURES)


print("COMBINED TF-IDF + RESEARCH MODEL")
print("================================")

print("Training samples:", len(train_text))
print("Validation samples:", len(val_text))


# ==================================================
# CREATE / VERIFY SAMPLE IDs
# ==================================================

# Training text file may not contain sample_id.
# If missing, recreate it using row order.

if "sample_id" not in train_text.columns:
    train_text.insert(
        0,
        "sample_id",
        range(len(train_text))
    )


# Validation text file already contains sample_id
# in the current dataset.
# Create it only if it is missing.

if "sample_id" not in val_text.columns:
    val_text.insert(
        0,
        "sample_id",
        range(len(val_text))
    )


# ==================================================
# VERIFY SAMPLE ALIGNMENT
# ==================================================

assert (
    train_text["sample_id"].tolist()
    == train_features["sample_id"].tolist()
), "Training feature rows are not aligned!"


assert (
    val_text["sample_id"].tolist()
    == val_features["sample_id"].tolist()
), "Validation feature rows are not aligned!"


print("Sample alignment: OK")


# ==================================================
# VERIFY LABEL ALIGNMENT
# ==================================================

assert (
    train_text["label"].tolist()
    == train_features["label"].tolist()
), "Training labels are not aligned!"


assert (
    val_text["label"].tolist()
    == val_features["label"].tolist()
), "Validation labels are not aligned!"


print("Label alignment: OK")


# ==================================================
# LABELS
# ==================================================

y_train = train_text["label"].values
y_val = val_text["label"].values


# ==================================================
# TF-IDF FEATURES
# ==================================================

print("\nBuilding TF-IDF features...")


vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)


# Fit vocabulary ONLY on training data

X_train_tfidf = vectorizer.fit_transform(
    train_text["text"].fillna("")
)


# Transform validation using training vocabulary

X_val_tfidf = vectorizer.transform(
    val_text["text"].fillna("")
)


print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)


print(
    "TF-IDF validation shape:",
    X_val_tfidf.shape
)


# ==================================================
# RESEARCH FEATURES
# ==================================================

research_columns = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity",
    "perplexity",
    "burstiness"
]


train_research_raw = train_features[
    research_columns
].copy()


val_research_raw = val_features[
    research_columns
].copy()


# ==================================================
# BURSTINESS MISSING INDICATOR
# ==================================================

missing_train = (
    train_research_raw["burstiness"]
    .isna()
    .astype(float)
    .values
    .reshape(-1, 1)
)


missing_val = (
    val_research_raw["burstiness"]
    .isna()
    .astype(float)
    .values
    .reshape(-1, 1)
)


# ==================================================
# LOG TRANSFORMATION
# ==================================================

train_research_raw["perplexity"] = np.log1p(
    train_research_raw["perplexity"]
)


train_research_raw["burstiness"] = np.log1p(
    train_research_raw["burstiness"]
)


val_research_raw["perplexity"] = np.log1p(
    val_research_raw["perplexity"]
)


val_research_raw["burstiness"] = np.log1p(
    val_research_raw["burstiness"]
)


# ==================================================
# MISSING VALUE IMPUTATION
# ==================================================

imputer = SimpleImputer(
    strategy="median"
)


# Fit ONLY on training data

train_research = imputer.fit_transform(
    train_research_raw
)


# Apply training-derived imputation to validation

val_research = imputer.transform(
    val_research_raw
)


# ==================================================
# STANDARDIZATION
# ==================================================

scaler = StandardScaler()


# Fit ONLY on training data

train_research = scaler.fit_transform(
    train_research
)


# Apply training-derived scaling to validation

val_research = scaler.transform(
    val_research
)


# ==================================================
# ADD BURSTINESS MISSING INDICATOR
# ==================================================

train_research = np.hstack(
    [
        train_research,
        missing_train
    ]
)


val_research = np.hstack(
    [
        val_research,
        missing_val
    ]
)


print(
    "Research feature shape:",
    train_research.shape
)


# ==================================================
# COMBINE TF-IDF + RESEARCH FEATURES
# ==================================================

X_train_combined = hstack(
    [
        X_train_tfidf,
        csr_matrix(train_research)
    ]
).tocsr()


X_val_combined = hstack(
    [
        X_val_tfidf,
        csr_matrix(val_research)
    ]
).tocsr()


print(
    "Combined training shape:",
    X_train_combined.shape
)


print(
    "Combined validation shape:",
    X_val_combined.shape
)


# ==================================================
# TRAIN CLASSIFIER
# ==================================================

print("\nTraining combined model...")


classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)


classifier.fit(
    X_train_combined,
    y_train
)


# ==================================================
# VALIDATION PREDICTION
# ==================================================

print(
    "Evaluating on unseen validation data..."
)


predictions = classifier.predict(
    X_val_combined
)


# ==================================================
# CALCULATE METRICS
# ==================================================

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


# ==================================================
# RESULTS
# ==================================================

print("\nCOMBINED MODEL RESULTS")
print("======================")


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


# ==================================================
# CONFUSION MATRIX
# ==================================================

print("\nCONFUSION MATRIX")
print("================")


cm = confusion_matrix(
    y_val,
    predictions
)


print(cm)


# ==================================================
# ADDITIONAL SUMMARY
# ==================================================

print("\nVALIDATION SUMMARY")
print("==================")


print(
    "Correct predictions:",
    int((predictions == y_val).sum())
)


print(
    "Incorrect predictions:",
    int((predictions != y_val).sum())
)


print(
    "Total validation samples:",
    len(y_val)
)