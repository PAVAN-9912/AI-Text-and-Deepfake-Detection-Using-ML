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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_DIR = r"models\text_final_v2"

TEST_FILE = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_test.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("V2 MODEL — SHORT-TEXT EXTERNAL TEST")
print("=" * 60)
print()


# =========================================================
# HELPER
# =========================================================

def get_text_column(df):

    if "text" in df.columns:
        return "text"

    if "generation" in df.columns:
        return "generation"

    raise ValueError(
        "No text column found."
    )


def clean_dataframe(df):

    text_column = get_text_column(df)

    df = df.copy()

    df["text"] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"].str.len() > 0
    ].copy()

    return df


def get_length_group(text):

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


# =========================================================
# LOAD TEST DATA
# =========================================================

print("LOADING SHORT-TEXT TEST DATA")
print("============================")

test_df = pd.read_csv(
    TEST_FILE
)

test_df = clean_dataframe(
    test_df
)

test_df["length_group"] = (
    test_df["text"]
    .apply(get_length_group)
)

print(
    f"External test samples: "
    f"{len(test_df)}"
)


# =========================================================
# LABEL PREPARATION
# =========================================================

if "label" in test_df.columns:

    y_true = (
        test_df["label"]
        .astype(int)
        .to_numpy()
    )

elif "label_name" in test_df.columns:

    y_true = (
        test_df["label_name"]
        .astype(str)
        .str.lower()
        .map({
            "human": 0,
            "ai": 1
        })
        .to_numpy()
    )

else:

    raise ValueError(
        "No label column found."
    )


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    pd.Series(
        y_true
    )
    .map({
        0: "Human",
        1: "AI"
    })
    .value_counts()
    .to_string()
)


print()
print("LENGTH DISTRIBUTION")
print("===================")

print(
    pd.crosstab(
        test_df["length_group"],
        pd.Series(
            y_true,
            index=test_df.index,
            name="label"
        )
        .map({
            0: "Human",
            1: "AI"
        })
    )
    .to_string()
)


# =========================================================
# LOAD V2 MODEL COMPONENTS
# =========================================================

print()
print("LOADING V2 TF-IDF MODELS")
print("========================")


with open(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    word_vectorizer = pickle.load(f)


with open(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    char_vectorizer = pickle.load(f)


with open(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    ),
    "rb"
) as f:

    linguistic_scaler = pickle.load(f)


with open(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    ),
    "rb"
) as f:

    classifier = pickle.load(f)


print(
    "Word vocabulary:",
    len(
        word_vectorizer.vocabulary_
    )
)

print(
    "Character vocabulary:",
    len(
        char_vectorizer.vocabulary_
    )
)


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")

word_features = (
    word_vectorizer.transform(
        test_df["text"]
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
print("BUILDING CHARACTER FEATURES")
print("============================")

char_features = (
    char_vectorizer.transform(
        test_df["text"]
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
print("LOADING MINILM")
print("==============")

print(
    "Transformer:",
    TRANSFORMER_NAME
)

model = SentenceTransformer(
    TRANSFORMER_NAME
)

print(
    "MiniLM loaded successfully."
)


# =========================================================
# EMBEDDINGS
# =========================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")

embeddings = model.encode(
    test_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    "Embedding shape:",
    embeddings.shape
)


embedding_features = csr_matrix(
    embeddings
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")

linguistic_features = (
    build_linguistic_features(
        test_df["text"].tolist()
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

print(
    "Scaled linguistic shape:",
    linguistic_features.shape
)


linguistic_features = csr_matrix(
    linguistic_features
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("COMBINING FEATURES")
print("==================")

combined_features = hstack(
    [
        word_features,
        char_features,
        embedding_features,
        linguistic_features
    ],
    format="csr"
)

print(
    "Combined feature shape:",
    combined_features.shape
)


# =========================================================
# VALIDATE FEATURE COUNT
# =========================================================

expected_features = (
    word_features.shape[1]
    +
    char_features.shape[1]
    +
    embedding_features.shape[1]
    +
    linguistic_features.shape[1]
)

print()
print("FEATURE VALIDATION")
print("==================")

print(
    "Expected features:",
    expected_features
)

print(
    "Classifier features:",
    classifier.n_features_in_
)


if expected_features != classifier.n_features_in_:

    raise RuntimeError(
        "Feature count mismatch!"
    )

print(
    "Feature configuration verified."
)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("RUNNING V2 PREDICTIONS")
print("======================")

probabilities = (
    classifier.predict_proba(
        combined_features
    )[:, 1]
)

predictions = (
    probabilities >= 0.50
).astype(int)


# =========================================================
# OVERALL RESULTS
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

roc_auc = roc_auc_score(
    y_true,
    probabilities
)


print()
print("=" * 60)
print("V2 SHORT-TEXT EXTERNAL RESULTS")
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
    f"ROC-AUC  : {roc_auc:.4f}"
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
print("PERFORMANCE BY TEXT LENGTH")
print("==========================")

for length_group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    mask = (
        test_df["length_group"]
        == length_group
    )

    if mask.sum() == 0:
        continue

    group_y = y_true[mask.to_numpy()]
    group_pred = predictions[mask.to_numpy()]

    group_accuracy = accuracy_score(
        group_y,
        group_pred
    )

    group_precision = precision_score(
        group_y,
        group_pred,
        zero_division=0
    )

    group_recall = recall_score(
        group_y,
        group_pred,
        zero_division=0
    )

    group_f1 = f1_score(
        group_y,
        group_pred,
        zero_division=0
    )

    print(
        f"{length_group:12s} "
        f"Accuracy={group_accuracy:.4f} "
        f"Precision={group_precision:.4f} "
        f"Recall={group_recall:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={mask.sum()}"
    )


# =========================================================
# AI RECALL BY LENGTH
# =========================================================

print()
print("AI DETECTION BY TEXT LENGTH")
print("===========================")

for length_group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    mask = (
        (test_df["length_group"] == length_group)
        &
        (y_true == 1)
    )

    if mask.sum() == 0:
        continue

    ai_recall = recall_score(
        y_true[mask.to_numpy()],
        predictions[mask.to_numpy()],
        zero_division=0
    )

    print(
        f"{length_group:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={mask.sum()}"
    )


# =========================================================
# SAVE DETAILED RESULTS
# =========================================================

results_df = test_df.copy()

results_df["true_label"] = y_true

results_df["predicted_label"] = predictions

results_df["true_label_name"] = (
    results_df["true_label"]
    .map({
        0: "Human",
        1: "AI"
    })
)

results_df["predicted_label_name"] = (
    results_df["predicted_label"]
    .map({
        0: "Human",
        1: "AI"
    })
)

results_df["ai_probability"] = probabilities

results_df["human_probability"] = (
    1.0 - probabilities
)


output_file = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_v2_final_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print()
print("RESULTS SAVED")
print("=============")

print(
    "Output:",
    output_file
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("V2 SHORT-TEXT FINAL SUMMARY")
print("=" * 60)

print(
    f"Total samples        : {len(y_true)}"
)

print(
    f"Correct predictions  : "
    f"{(y_true == predictions).sum()}"
)

print(
    f"Incorrect predictions: "
    f"{(y_true != predictions).sum()}"
)

print(
    f"Accuracy             : {accuracy:.4f}"
)

print(
    f"F1 Score             : {f1:.4f}"
)

print(
    f"ROC-AUC              : {roc_auc:.4f}"
)

print()
print(
    "V2 SHORT-TEXT EXTERNAL "
    "EVALUATION COMPLETED"
)