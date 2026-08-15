import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_TEXT = r"datasets\text\HC3\feature_development.csv"
TRAIN_FEATURES = r"datasets\text\HC3\research_features.csv"

VAL_TEXT = r"datasets\text\HC3\feature_validation.csv"
VAL_FEATURES = r"datasets\text\HC3\validation_research_features.csv"

MODEL_DIR = r"models\text"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("FINAL TEXT MODEL TRAINING")
print("=========================")

train_text = pd.read_csv(TRAIN_TEXT)
train_features = pd.read_csv(TRAIN_FEATURES)

val_text = pd.read_csv(VAL_TEXT)
val_features = pd.read_csv(VAL_FEATURES)

print("Training samples:", len(train_text))
print("Validation samples:", len(val_text))


# ============================================================
# SAMPLE IDs
# ============================================================

if "sample_id" not in train_text.columns:
    train_text.insert(
        0,
        "sample_id",
        range(len(train_text))
    )

if "sample_id" not in val_text.columns:
    val_text.insert(
        0,
        "sample_id",
        range(len(val_text))
    )


# ============================================================
# ALIGNMENT CHECK
# ============================================================

assert (
    train_text["sample_id"].tolist()
    == train_features["sample_id"].tolist()
), "Training feature rows are not aligned!"

assert (
    val_text["sample_id"].tolist()
    == val_features["sample_id"].tolist()
), "Validation feature rows are not aligned!"

print("Sample alignment: OK")


# ============================================================
# LABEL CHECK
# ============================================================

assert (
    train_text["label"].tolist()
    == train_features["label"].tolist()
), "Training labels are not aligned!"

assert (
    val_text["label"].tolist()
    == val_features["label"].tolist()
), "Validation labels are not aligned!"

print("Label alignment: OK")


y_train = train_text["label"].values
y_val = val_text["label"].values


# ============================================================
# TF-IDF
# ============================================================

print("\nBuilding TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(
    train_text["text"].fillna("")
)

X_val_tfidf = vectorizer.transform(
    val_text["text"].fillna("")
)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF validation shape:", X_val_tfidf.shape)


# ============================================================
# RESEARCH FEATURES
# ============================================================

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


# ============================================================
# BURSTINESS MISSING INDICATOR
# ============================================================

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


# ============================================================
# LOG TRANSFORMATION
# ============================================================

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


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

train_research = imputer.fit_transform(
    train_research_raw
)

val_research = imputer.transform(
    val_research_raw
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

train_research = scaler.fit_transform(
    train_research
)

val_research = scaler.transform(
    val_research
)


# ============================================================
# ADD MISSING INDICATOR
# ============================================================

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

print("Research feature shape:", train_research.shape)


# ============================================================
# COMBINE FEATURES
# ============================================================

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

print("Combined training shape:", X_train_combined.shape)
print("Combined validation shape:", X_val_combined.shape)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

classifier.fit(
    X_train_combined,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

print("\nEvaluating validation set...")

predictions = classifier.predict(
    X_val_combined
)

probabilities = classifier.predict_proba(
    X_val_combined
)[:, 1]


accuracy = accuracy_score(
    y_val,
    predictions
)

precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_val,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\nFINAL TEXT MODEL RESULTS")
print("========================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nCONFUSION MATRIX")
print("================")
print(cm)

print("\nCLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_val,
        predictions,
        target_names=["Human", "AI"],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

print("\nSAVING MODEL ARTIFACTS...")
print("=========================")

joblib.dump(
    vectorizer,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

joblib.dump(
    imputer,
    os.path.join(
        MODEL_DIR,
        "research_imputer.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "research_scaler.pkl"
    )
)

joblib.dump(
    classifier,
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


# Save configuration for inference

config = {
    "research_columns": research_columns,
    "tfidf_features": X_train_tfidf.shape[1],
    "research_features": train_research.shape[1],
    "total_features": X_train_combined.shape[1],
    "model_type": "TF-IDF + Research Features + Logistic Regression",
    "label_mapping": {
        "0": "Human",
        "1": "AI"
    }
}

joblib.dump(
    config,
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


# ============================================================
# SAVE VALIDATION METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "validation_samples": int(len(y_val))
}

joblib.dump(
    metrics,
    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\nMODEL SAVED")
print("===========")

print(
    "Directory:",
    MODEL_DIR
)

print("Files:")

for filename in [
    "tfidf_vectorizer.pkl",
    "research_imputer.pkl",
    "research_scaler.pkl",
    "text_classifier.pkl",
    "text_config.pkl",
    "text_metrics.pkl"
]:
    print(" -", filename)

print("\nTEXT MODEL TRAINING COMPLETED")