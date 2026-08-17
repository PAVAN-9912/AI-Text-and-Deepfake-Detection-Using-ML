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


# ============================================================
# V2 IMPROVED — HC3 HELD-OUT EVALUATION
# ============================================================

print()
print("=" * 60)
print("V2 IMPROVED MODEL — HC3 HELD-OUT VALIDATION")
print("=" * 60)


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = r"models\text_final_v2_improved"

HC3_VALIDATION = r"datasets\text\HC3\validation.csv"

OUTPUT_FILE = (
    r"datasets\text\HC3"
    r"\hc3_v2_improved_validation_results.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)


def get_text_column(df):

    if "text" in df.columns:
        return "text"

    if "generation" in df.columns:
        return "generation"

    raise ValueError(
        "No text column found in HC3 validation dataset."
    )


def clean_dataframe(df):

    df = df.copy()

    text_column = get_text_column(df)

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
        str(text).split()
    )

    if word_count <= 10:
        return "very_short"

    elif word_count <= 30:
        return "short"

    elif word_count <= 60:
        return "medium"

    else:
        return "long"


def add_length_group(df):

    df = df.copy()

    df["length_group"] = (
        df["text"]
        .apply(get_length_group)
    )

    return df


def build_linguistic_features(texts):

    features = []

    for text in texts:

        text = str(text)

        words = text.split()

        word_count = len(words)

        # ----------------------------------------------------
        # Sentence count
        # ----------------------------------------------------

        sentence_count = sum(
            1
            for char in text
            if char in ".!?"
        )

        if sentence_count == 0:
            sentence_count = 1

        # ----------------------------------------------------
        # Average sentence length
        # ----------------------------------------------------

        avg_sentence_length = (
            word_count /
            max(sentence_count, 1)
        )

        # ----------------------------------------------------
        # Vocabulary diversity
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Log word count
        # ----------------------------------------------------

        log_word_count = np.log1p(
            word_count
        )

        # ----------------------------------------------------
        # Length bucket
        # ----------------------------------------------------

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


# ============================================================
# LOAD HC3
# ============================================================

print()
print("LOADING HC3 VALIDATION")
print("======================")

hc3 = pd.read_csv(
    HC3_VALIDATION
)

hc3 = clean_dataframe(hc3)

hc3 = add_length_group(hc3)


# ============================================================
# NORMALIZE LABELS
# ============================================================

if hc3["label"].dtype == object:

    hc3["label"] = (
        hc3["label"]
        .map(
            {
                "Human": 0,
                "AI": 1,
                "human": 0,
                "ai": 1
            }
        )
    )


hc3 = hc3[
    hc3["label"].isin([0, 1])
].copy()


hc3["label"] = (
    hc3["label"]
    .astype(int)
)


print(
    "Validation samples:",
    len(hc3)
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    hc3["label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
    .value_counts()
    .to_string()
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

if "source" in hc3.columns:

    print()
    print("SOURCE DISTRIBUTION")
    print("===================")

    print(
        hc3["source"]
        .value_counts()
        .to_string()
    )


# ============================================================
# LENGTH DISTRIBUTION
# ============================================================

print()
print("TEXT LENGTH DISTRIBUTION")
print("========================")

print(
    pd.crosstab(
        hc3["length_group"],
        hc3["label"]
    )
    .rename(
        columns={
            0: "Human",
            1: "AI"
        }
    )
    .to_string()
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("LOADING V2 IMPROVED MODEL")
print("==========================")


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


print(
    "V2 Improved model components loaded."
)


expected_features = config.get(
    "feature_count"
)


print(
    "Expected feature count:",
    expected_features
)


# ============================================================
# DECISION THRESHOLD
# ============================================================

decision_threshold = config.get(
    "decision_threshold",
    0.50
)


print(
    f"V2 Improved decision threshold: "
    f"{decision_threshold:.2f}"
)


# ============================================================
# WORD TF-IDF
# ============================================================

print()
print("BUILDING WORD FEATURES")
print("======================")

X_word = (
    word_vectorizer.transform(
        hc3["text"]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        hc3["text"]
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# ============================================================
# MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


transformer_name = config.get(
    "transformer",
    "sentence-transformers/all-MiniLM-L6-v2"
)


print(
    "Transformer:",
    transformer_name
)


embedder = SentenceTransformer(
    transformer_name
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# EMBEDDINGS
# ============================================================

print()
print("CREATING MINILM EMBEDDINGS")
print("===========================")


X_embedding = embedder.encode(
    hc3["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    X_embedding.shape
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_linguistic_raw = (
    build_linguistic_features(
        hc3["text"].tolist()
    )
)


print(
    "Raw linguistic shape:",
    X_linguistic_raw.shape
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic_raw
    )
)


print(
    "Scaled linguistic shape:",
    X_linguistic.shape
)


# ============================================================
# SPARSE CONVERSION
# ============================================================

X_embedding_sparse = csr_matrix(
    X_embedding
)


X_linguistic_sparse = csr_matrix(
    X_linguistic
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


X = hstack(
    [
        X_word,
        X_char,
        X_embedding_sparse,
        X_linguistic_sparse
    ],
    format="csr"
)


print(
    "Combined feature shape:",
    X.shape
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print()
print("FEATURE VALIDATION")
print("==================")


actual_features = X.shape[1]

classifier_features = (
    classifier.n_features_in_
)


print(
    "Expected features:",
    expected_features
)

print(
    "Classifier features:",
    classifier_features
)

print(
    "Actual features:",
    actual_features
)


if (
    expected_features !=
    classifier_features
    or
    actual_features !=
    classifier_features
):

    raise RuntimeError(
        "Feature configuration mismatch."
    )


print(
    "Feature configuration verified."
)


# ============================================================
# LABELS
# ============================================================

y_true = (
    hc3["label"]
    .astype(int)
    .values
)


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

print()
print("RUNNING V2 IMPROVED PREDICTIONS")
print("===============================")


probabilities = (
    classifier.predict_proba(
        X
    )[:, 1]
)


predictions = (
    probabilities >=
    decision_threshold
).astype(int)


# ============================================================
# OVERALL METRICS
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("V2 IMPROVED HC3 HELD-OUT RESULTS")
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

print(
    f"Threshold: {decision_threshold:.2f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")


cm = confusion_matrix(
    y_true,
    predictions
)


print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# SAVE PREDICTIONS
# ============================================================

result_df = hc3.copy()


result_df["prediction"] = (
    predictions
)


result_df["probability"] = (
    probabilities
)


result_df["predicted_label"] = (
    np.where(
        predictions == 1,
        "AI",
        "Human"
    )
)


# ============================================================
# PERFORMANCE BY LENGTH
# ============================================================

print()
print("PERFORMANCE BY TEXT LENGTH")
print("===========================")


length_results = []


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = result_df[
        result_df["length_group"]
        == group
    ]


    if len(subset) == 0:
        continue


    group_accuracy = accuracy_score(
        subset["label"],
        subset["prediction"]
    )


    group_precision = precision_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    group_recall = recall_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    group_f1 = f1_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    print(
        f"{group:12s} "
        f"Accuracy={group_accuracy:.4f} "
        f"Precision={group_precision:.4f} "
        f"Recall={group_recall:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={len(subset)}"
    )


    length_results.append(
        {
            "length_group": group,
            "accuracy": group_accuracy,
            "precision": group_precision,
            "recall": group_recall,
            "f1": group_f1,
            "samples": len(subset)
        }
    )


# ============================================================
# AI RECALL BY LENGTH
# ============================================================

print()
print("AI DETECTION BY TEXT LENGTH")
print("===========================")


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = result_df[
        (
            result_df["length_group"]
            == group
        )
        &
        (
            result_df["label"]
            == 1
        )
    ]


    if len(subset) == 0:
        continue


    ai_recall = (
        (subset["prediction"] == 1)
        .mean()
    )


    print(
        f"{group:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={len(subset)}"
    )


# ============================================================
# PERFORMANCE BY HC3 SOURCE
# ============================================================

if "source" in result_df.columns:

    print()
    print("PERFORMANCE BY HC3 SOURCE")
    print("=========================")


    source_results = []


    for source, subset in (
        result_df.groupby("source")
    ):

        source_accuracy = (
            accuracy_score(
                subset["label"],
                subset["prediction"]
            )
        )


        source_f1 = (
            f1_score(
                subset["label"],
                subset["prediction"],
                zero_division=0
            )
        )


        source_recall = (
            recall_score(
                subset["label"],
                subset["prediction"],
                zero_division=0
            )
        )


        print(
            f"{str(source):20s} "
            f"Accuracy={source_accuracy:.4f} "
            f"Recall={source_recall:.4f} "
            f"F1={source_f1:.4f} "
            f"Samples={len(subset)}"
        )


        source_results.append(
            {
                "source": source,
                "accuracy": source_accuracy,
                "recall": source_recall,
                "f1": source_f1,
                "samples": len(subset)
            }
        )


# ============================================================
# ERROR ANALYSIS
# ============================================================

false_positives = (
    (
        result_df["label"] == 0
    )
    &
    (
        result_df["prediction"] == 1
    )
).sum()


false_negatives = (
    (
        result_df["label"] == 1
    )
    &
    (
        result_df["prediction"] == 0
    )
).sum()


print()
print("ERROR ANALYSIS")
print("==============")


print(
    "Human predicted as AI (FP):",
    false_positives
)


print(
    "AI predicted as Human (FN):",
    false_negatives
)


# ============================================================
# SAVE RESULTS
# ============================================================

print()
print("RESULTS SAVED")
print("=============")


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("V2 IMPROVED HC3 FINAL SUMMARY")
print("=" * 60)


print(
    f"Total samples        : {len(result_df)}"
)


print(
    f"Correct predictions  : "
    f"{int((predictions == y_true).sum())}"
)


print(
    f"Incorrect predictions: "
    f"{int((predictions != y_true).sum())}"
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


print(
    f"AI Recall            : {recall:.4f}"
)


print()
print(
    "V2 IMPROVED HC3 HELD-OUT "
    "EVALUATION COMPLETED"
)