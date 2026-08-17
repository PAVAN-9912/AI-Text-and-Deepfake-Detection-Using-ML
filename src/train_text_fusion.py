import os
import re
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = r"datasets\text\HC3\train.csv"
VALIDATION_PATH = r"datasets\text\HC3\validation.csv"

MODEL_DIR = r"models\text_fusion"

MINILM_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("TEXT FUSION DETECTOR")
print("====================")

print(
    "Transformer:",
    MINILM_NAME
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("LOADING DATA")
print("============")


train_df = pd.read_csv(
    TRAIN_PATH
)


validation_df = pd.read_csv(
    VALIDATION_PATH
)


print(
    "Training samples:",
    len(train_df)
)


print(
    "Validation samples:",
    len(validation_df)
)


print()
print("TRAIN LABEL DISTRIBUTION")
print("=========================")

print(
    train_df["label"].value_counts()
)


print()
print("VALIDATION LABEL DISTRIBUTION")
print("=============================")

print(
    validation_df["label"].value_counts()
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


train_texts = (
    train_df["text"]
    .fillna("")
    .map(clean_text)
    .tolist()
)


validation_texts = (
    validation_df["text"]
    .fillna("")
    .map(clean_text)
    .tolist()
)


y_train = (
    train_df["label"]
    .astype(int)
    .to_numpy()
)


y_validation = (
    validation_df["label"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# SOURCE-BALANCED SAMPLE
#
# HC3 is heavily dominated by reddit_eli5.
# We keep a balanced subset from every source.
# ============================================================

print()
print("BUILDING SOURCE-BALANCED TRAINING SET")
print("======================================")


balanced_parts = []


for source in sorted(
    train_df["source"].dropna().unique()
):

    source_df = train_df[
        train_df["source"] == source
    ]


    human_df = source_df[
        source_df["label"] == 0
    ]


    ai_df = source_df[
        source_df["label"] == 1
    ]


    count = min(
        len(human_df),
        len(ai_df)
    )


    # Keep at most 5000 examples
    # from each class for each source.
    count = min(
        count,
        5000
    )


    human_sample = human_df.sample(
        n=count,
        random_state=42
    )


    ai_sample = ai_df.sample(
        n=count,
        random_state=42
    )


    balanced_parts.append(
        human_sample
    )

    balanced_parts.append(
        ai_sample
    )


    print(
        f"{source:15s} "
        f"Human={count:5d} "
        f"AI={count:5d}"
    )


balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True
)


balanced_train = balanced_train.sample(
    frac=1.0,
    random_state=42
).reset_index(
    drop=True
)


print()
print(
    "Balanced training samples:",
    len(balanced_train)
)


print()
print("BALANCED LABEL DISTRIBUTION")
print("===========================")

print(
    balanced_train["label"].value_counts()
)


train_texts = (
    balanced_train["text"]
    .fillna("")
    .map(clean_text)
    .tolist()
)


y_train = (
    balanced_train["label"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# WORD TF-IDF
# ============================================================

print()
print("BUILDING WORD TF-IDF")
print("====================")


word_vectorizer = TfidfVectorizer(

    analyzer="word",

    ngram_range=(
        1,
        2
    ),

    min_df=2,

    max_df=0.98,

    max_features=150000,

    sublinear_tf=True,

    strip_accents="unicode"
)


print(
    "Fitting word vocabulary..."
)


X_word_train = (
    word_vectorizer.fit_transform(
        train_texts
    )
)


print(
    "Transforming validation..."
)


X_word_validation = (
    word_vectorizer.transform(
        validation_texts
    )
)


print(
    "Training word shape:",
    X_word_train.shape
)


print(
    "Validation word shape:",
    X_word_validation.shape
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print()
print("BUILDING CHARACTER TF-IDF")
print("=========================")


char_vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(
        3,
        5
    ),

    min_df=2,

    max_features=150000,

    sublinear_tf=True
)


print(
    "Fitting character vocabulary..."
)


X_char_train = (
    char_vectorizer.fit_transform(
        train_texts
    )
)


print(
    "Transforming validation..."
)


X_char_validation = (
    char_vectorizer.transform(
        validation_texts
    )
)


print(
    "Training character shape:",
    X_char_train.shape
)


print(
    "Validation character shape:",
    X_char_validation.shape
)


# ============================================================
# MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


minilm = SentenceTransformer(
    MINILM_NAME
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    texts,
    name
):

    print()
    print(
        "CREATING",
        name.upper(),
        "EMBEDDINGS"
    )


    embeddings = minilm.encode(

        texts,

        batch_size=16,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )


    return embeddings


X_minilm_train = create_embeddings(
    train_texts,
    "training"
)


X_minilm_validation = create_embeddings(
    validation_texts,
    "validation"
)


print()
print("MiniLM training shape:")

print(
    X_minilm_train.shape
)


print(
    "MiniLM validation shape:"
)

print(
    X_minilm_validation.shape
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

    text = str(text)


    words = re.findall(
        r"\b\w+\b",
        text
    )


    word_count = len(
        words
    )


    sentence_count = max(
        len(
            re.findall(
                r"[.!?]+",
                text
            )
        ),
        1
    )


    avg_sentence_length = (
        word_count /
        sentence_count
    )


    unique_words = len(
        set(
            word.lower()
            for word in words
        )
    )


    vocabulary_diversity = (
        unique_words /
        max(
            word_count,
            1
        )
    )


    return [

        word_count,

        sentence_count,

        avg_sentence_length,

        vocabulary_diversity

    ]


print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_linguistic_train = np.array(
    [
        linguistic_features(text)
        for text in train_texts
    ],
    dtype=float
)


X_linguistic_validation = np.array(
    [
        linguistic_features(text)
        for text in validation_texts
    ],
    dtype=float
)


linguistic_scaler = (
    StandardScaler()
)


X_linguistic_train = (
    linguistic_scaler.fit_transform(
        X_linguistic_train
    )
)


X_linguistic_validation = (
    linguistic_scaler.transform(
        X_linguistic_validation
    )
)


print(
    "Training linguistic shape:",
    X_linguistic_train.shape
)


print(
    "Validation linguistic shape:",
    X_linguistic_validation.shape
)


# ============================================================
# COMBINE ALL FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


# TF-IDF is sparse.
# MiniLM + linguistic features are dense.
#
# To avoid unnecessarily converting the huge TF-IDF
# matrices to dense arrays, only the small MiniLM
# and linguistic matrices are converted to sparse.

X_dense_train = np.hstack(
    [
        X_minilm_train,
        X_linguistic_train
    ]
)


X_dense_validation = np.hstack(
    [
        X_minilm_validation,
        X_linguistic_validation
    ]
)


X_combined_train = hstack(
    [
        X_word_train,
        X_char_train,
        csr_matrix(
            X_dense_train
        )
    ]
).tocsr()


X_combined_validation = hstack(
    [
        X_word_validation,
        X_char_validation,
        csr_matrix(
            X_dense_validation
        )
    ]
).tocsr()


print(
    "Combined training shape:",
    X_combined_train.shape
)


print(
    "Combined validation shape:",
    X_combined_validation.shape
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("TRAINING FUSION CLASSIFIER")
print("==========================")


classifier = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    C=2.0,

    solver="liblinear",

    random_state=42
)


classifier.fit(
    X_combined_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("EVALUATING VALIDATION SET")
print("=========================")


predictions = classifier.predict(
    X_combined_validation
)


probabilities = (
    classifier.predict_proba(
        X_combined_validation
    )[:, 1]
)


accuracy = accuracy_score(
    y_validation,
    predictions
)


precision = precision_score(
    y_validation,
    predictions,
    zero_division=0
)


recall = recall_score(
    y_validation,
    predictions,
    zero_division=0
)


f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_validation,
    probabilities
)


cm = confusion_matrix(
    y_validation,
    predictions
)


print()
print("TEXT FUSION MODEL RESULTS")
print("=========================")


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
    f"ROC-AUC  : {roc_auc:.4f}"
)


print()
print("CONFUSION MATRIX")
print("================")

print(
    cm
)


print()
print("CLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_validation,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("SAVING FUSION MODEL")
print("===================")


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
    linguistic_scaler,
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)


joblib.dump(
    classifier,
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


joblib.dump(

    {
        "model_type":
            "word_char_minilm_linguistic_fusion",

        "minilm_name":
            MINILM_NAME,

        "word_features":
            X_word_train.shape[1],

        "character_features":
            X_char_train.shape[1],

        "minilm_features":
            X_minilm_train.shape[1],

        "linguistic_features":
            X_linguistic_train.shape[1],

        "total_features":
            X_combined_train.shape[1],

        "validation_accuracy":
            accuracy,

        "validation_precision":
            precision,

        "validation_recall":
            recall,

        "validation_f1":
            f1,

        "validation_roc_auc":
            roc_auc

    },

    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


joblib.dump(

    {
        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            roc_auc,

        "confusion_matrix":
            cm.tolist()

    },

    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    )
)


print()
print("FUSION MODEL SAVED")
print("==================")

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
    " - linguistic_scaler.pkl"
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
    "TEXT FUSION MODEL TRAINING COMPLETED"
)