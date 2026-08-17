import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sentence_transformers import SentenceTransformer


# =========================================================
# V3 REAL-WORLD EVALUATION
# =========================================================

print()
print("=" * 60)
print("V3 MODEL — REAL-WORLD TEXT TEST")
print("=" * 60)


# =========================================================
# PATHS
# =========================================================

MODEL_DIR = r"models\text_final_v3"

TEST_FILE = (
    r"datasets\text\realworld_test.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_pickle(path):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(f)


def build_linguistic_features(texts):

    features = []

    for text in texts:

        words = text.split()

        word_count = len(words)

        sentence_count = sum(
            1
            for char in text
            if char in ".!?"
        )

        if sentence_count == 0:
            sentence_count = 1

        avg_sentence_length = (
            word_count /
            max(sentence_count, 1)
        )

        if word_count > 0:

            unique_words = len(
                set(
                    word.lower()
                    for word in words
                )
            )

            vocabulary_diversity = (
                unique_words /
                word_count
            )

        else:

            vocabulary_diversity = 0.0

        log_word_count = np.log1p(
            word_count
        )

        if word_count <= 10:
            length_bucket = 0.0

        elif word_count <= 30:
            length_bucket = 1.0

        elif word_count <= 60:
            length_bucket = 2.0

        else:
            length_bucket = 3.0

        features.append(
            [
                log_word_count,
                sentence_count,
                avg_sentence_length,
                vocabulary_diversity,
                length_bucket
            ]
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


def classify_length(text):

    word_count = len(
        text.split()
    )

    if word_count <= 10:
        return "very_short"

    elif word_count <= 30:
        return "short"

    elif word_count <= 60:
        return "medium"

    return "long"


# =========================================================
# LOAD REAL-WORLD DATA
# =========================================================

print()
print("LOADING REAL-WORLD TEST DATA")
print("============================")

df = pd.read_csv(
    TEST_FILE
)


df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df["length_group"] = (
    df["text"]
    .apply(classify_length)
)


print(
    f"Test samples: {len(df)}"
)


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "Human",
            1: "AI"
        }
    )
    .to_string()
)


print()
print("CATEGORY DISTRIBUTION")
print("======================")

print(
    df["category"]
    .value_counts()
    .to_string()
)


print()
print("LENGTH DISTRIBUTION")
print("===================")

print(
    pd.crosstab(
        df["length_group"],
        df["label"]
    )
    .reindex(
        [
            "very_short",
            "short",
            "medium",
            "long"
        ]
    )
    .fillna(0)
    .astype(int)
    .rename(
        columns={
            0: "Human",
            1: "AI"
        }
    )
    .to_string()
)


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("=" * 60)
print("LOADING V3 MODEL")
print("=" * 60)


word_vectorizer = load_pickle(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)

char_vectorizer = load_pickle(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)

linguistic_scaler = load_pickle(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)

classifier = load_pickle(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)

config = load_pickle(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


THRESHOLD = float(
    config.get(
        "threshold",
        0.50
    )
)


print(
    "V3 model components loaded."
)

print(
    f"Decision threshold: "
    f"{THRESHOLD:.2f}"
)


# =========================================================
# TEXT
# =========================================================

texts = (
    df["text"]
    .tolist()
)


y_true = (
    df["label"]
    .astype(int)
    .to_numpy()
)


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("=" * 60)
print("BUILDING WORD FEATURES")
print("=" * 60)


word_features = (
    word_vectorizer.transform(
        texts
    )
)


print(
    "Word feature shape:",
    word_features.shape
)


# =========================================================
# CHARACTER FEATURES
# =========================================================

print()
print("=" * 60)
print("BUILDING CHARACTER FEATURES")
print("=" * 60)


char_features = (
    char_vectorizer.transform(
        texts
    )
)


print(
    "Character feature shape:",
    char_features.shape
)


# =========================================================
# MINILM
# =========================================================

print()
print("=" * 60)
print("LOADING MINILM")
print("=" * 60)

print(
    "Transformer:",
    TRANSFORMER_NAME
)


transformer = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# =========================================================
# EMBEDDINGS
# =========================================================

print()
print("=" * 60)
print("CREATING MINILM EMBEDDINGS")
print("=" * 60)


embeddings = transformer.encode(

    texts,

    batch_size=32,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


print(
    "Embedding shape:",
    embeddings.shape
)


embeddings = csr_matrix(
    embeddings
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("=" * 60)
print("BUILDING LINGUISTIC FEATURES")
print("=" * 60)


linguistic_features = (
    build_linguistic_features(
        texts
    )
)


print(
    "Raw linguistic shape:",
    linguistic_features.shape
)


linguistic_features = (
    linguistic_scaler.transform(
        linguistic_features
    )
)


linguistic_features = csr_matrix(
    linguistic_features
)


print(
    "Scaled linguistic shape:",
    linguistic_features.shape
)


# =========================================================
# COMBINE
# =========================================================

print()
print("=" * 60)
print("COMBINING FEATURES")
print("=" * 60)


features = hstack(
    [
        word_features,
        char_features,
        embeddings,
        linguistic_features
    ],
    format="csr"
)


print(
    "Combined feature shape:",
    features.shape
)


# =========================================================
# FEATURE VALIDATION
# =========================================================

print()
print("=" * 60)
print("FEATURE VALIDATION")
print("=" * 60)


expected_features = int(
    config["feature_count"]
)

actual_features = (
    features.shape[1]
)

classifier_features = (
    classifier.n_features_in_
)


print(
    "Expected features:",
    expected_features
)

print(
    "Actual features:",
    actual_features
)

print(
    "Classifier features:",
    classifier_features
)


if (
    actual_features
    != classifier_features
):

    raise RuntimeError(
        "Feature count mismatch."
    )


print(
    "Feature configuration verified."
)


# =========================================================
# PREDICTION
# =========================================================

print()
print("=" * 60)
print("RUNNING V3 PREDICTIONS")
print("=" * 60)


probabilities = (
    classifier.predict_proba(
        features
    )[:, 1]
)


predictions = (
    probabilities
    >= THRESHOLD
).astype(int)


# =========================================================
# OVERALL METRICS
# =========================================================

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


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("V3 REAL-WORLD RESULTS")
print("=" * 60)

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
    f"Threshold: {THRESHOLD:.2f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print("================")

print(
    confusion_matrix(
        y_true,
        predictions
    )
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("CLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_true,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# =========================================================
# PERFORMANCE BY LENGTH
# =========================================================

print()
print("=" * 60)
print("PERFORMANCE BY TEXT LENGTH")
print("=" * 60)


for length_group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    mask = (
        df["length_group"]
        ==
        length_group
    )

    if mask.sum() == 0:
        continue


    true_group = (
        y_true[mask]
    )

    pred_group = (
        predictions[mask]
    )


    group_accuracy = (
        accuracy_score(
            true_group,
            pred_group
        )
    )

    group_precision = (
        precision_score(
            true_group,
            pred_group,
            zero_division=0
        )
    )

    group_recall = (
        recall_score(
            true_group,
            pred_group,
            zero_division=0
        )
    )

    group_f1 = (
        f1_score(
            true_group,
            pred_group,
            zero_division=0
        )
    )


    print(
        f"{length_group:11s} "
        f"Accuracy={group_accuracy:.4f} "
        f"Precision={group_precision:.4f} "
        f"Recall={group_recall:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={mask.sum()}"
    )


# =========================================================
# PERFORMANCE BY CATEGORY
# =========================================================

print()
print("=" * 60)
print("PERFORMANCE BY CATEGORY")
print("=" * 60)


category_rows = []


for category in sorted(
    df["category"].unique()
):

    mask = (
        df["category"]
        ==
        category
    )


    true_group = (
        y_true[mask]
    )

    pred_group = (
        predictions[mask]
    )


    category_accuracy = (
        accuracy_score(
            true_group,
            pred_group
        )
    )


    category_rows.append(
        {
            "category":
                category,

            "samples":
                int(mask.sum()),

            "correct":
                int(
                    (
                        true_group
                        ==
                        pred_group
                    ).sum()
                ),

            "accuracy":
                category_accuracy
        }
    )


    print(
        f"{category:20s} "
        f"Accuracy={category_accuracy:.4f} "
        f"Samples={mask.sum()}"
    )


# =========================================================
# INDIVIDUAL PREDICTIONS
# =========================================================

print()
print("=" * 60)
print("INDIVIDUAL PREDICTIONS")
print("=" * 60)


for i in range(
    len(df)
):

    actual = (
        "AI"
        if y_true[i] == 1
        else "Human"
    )

    predicted = (
        "AI"
        if predictions[i] == 1
        else "Human"
    )

    correct = (
        "YES"
        if actual == predicted
        else "NO"
    )

    print()
    print(
        f"[{i + 1}] "
        f"{df.iloc[i]['category']}"
    )

    print(
        f"Actual    : {actual}"
    )

    print(
        f"Prediction: {predicted}"
    )

    print(
        f"AI Prob.  : "
        f"{probabilities[i] * 100:.2f}%"
    )

    print(
        f"Correct   : {correct}"
    )

    print(
        f"Text      : "
        f"{df.iloc[i]['text']}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

df["ai_probability"] = (
    probabilities
)

df["human_probability"] = (
    1.0 -
    probabilities
)

df["prediction"] = (
    predictions
)

df["prediction_name"] = np.where(
    predictions == 1,
    "AI-Generated",
    "Human-Written"
)

df["correct"] = (
    predictions
    ==
    y_true
)


output_file = (
    r"datasets\text"
    r"\realworld_v3_results.csv"
)


df.to_csv(
    output_file,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

correct_count = int(
    df["correct"].sum()
)

incorrect_count = (
    len(df)
    -
    correct_count
)


print()
print("=" * 60)
print("V3 REAL-WORLD FINAL SUMMARY")
print("=" * 60)

print(
    f"Total samples        : {len(df)}"
)

print(
    f"Correct predictions  : {correct_count}"
)

print(
    f"Incorrect predictions: {incorrect_count}"
)

print(
    f"Accuracy             : "
    f"{accuracy:.4f}"
)

print(
    f"F1 Score             : "
    f"{f1:.4f}"
)

print()
print(
    "Results saved:"
)

print(
    output_file
)

print()
print("=" * 60)
print("V3 REAL-WORLD EVALUATION COMPLETED")
print("=" * 60)