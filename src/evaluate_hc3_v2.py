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

TEST_FILE = r"datasets\text\HC3\validation.csv"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("V2 MODEL — HC3 HELD-OUT VALIDATION")
print("=" * 60)
print()


# =========================================================
# HELPER FUNCTIONS
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

    word_count = len(text.split())

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
# LOAD HC3 VALIDATION
# =========================================================

print("LOADING HC3 VALIDATION")
print("======================")

df = pd.read_csv(
    TEST_FILE
)

df = clean_dataframe(df)

df["length_group"] = (
    df["text"]
    .apply(get_length_group)
)


print(
    f"Validation samples: {len(df)}"
)


# =========================================================
# LABELS
# =========================================================

if "label" in df.columns:

    y_true = (
        df["label"]
        .astype(int)
        .to_numpy()
    )

elif "label_name" in df.columns:

    y_true = (
        df["label_name"]
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
    pd.Series(y_true)
    .map({
        0: "Human",
        1: "AI"
    })
    .value_counts()
    .to_string()
)


# =========================================================
# DATASET DISTRIBUTION
# =========================================================

if "source" in df.columns:

    print()
    print("SOURCE DISTRIBUTION")
    print("===================")

    print(
        df["source"]
        .value_counts()
        .to_string()
    )


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("TEXT LENGTH DISTRIBUTION")
print("========================")

print(
    pd.crosstab(
        df["length_group"],
        pd.Series(
            y_true,
            index=df.index,
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
# LOAD V2 MODEL
# =========================================================

print()
print("LOADING V2 MODEL")
print("================")


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


print("V2 model components loaded.")


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")

word_features = (
    word_vectorizer.transform(
        df["text"]
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
        df["text"]
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
print("CREATING MINILM EMBEDDINGS")
print("===========================")

embeddings = transformer.encode(
    df["text"].tolist(),
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
        df["text"].tolist()
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
# COMBINE
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
# FEATURE VALIDATION
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
        "Feature count mismatch."
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

roc_auc = roc_auc_score(
    y_true,
    probabilities
)


print()
print("=" * 60)
print("V2 HC3 HELD-OUT RESULTS")
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
        df["length_group"]
        .to_numpy()
        ==
        length_group
    )

    if mask.sum() == 0:
        continue

    group_y = y_true[mask]

    group_pred = predictions[mask]

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
        (
            df["length_group"]
            .to_numpy()
            ==
            length_group
        )
        &
        (y_true == 1)
    )

    if mask.sum() == 0:
        continue

    ai_recall = recall_score(
        y_true[mask],
        predictions[mask],
        zero_division=0
    )

    print(
        f"{length_group:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={mask.sum()}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results = df.copy()

results["true_label"] = y_true

results["predicted_label"] = predictions

results["true_label_name"] = (
    results["true_label"]
    .map({
        0: "Human",
        1: "AI"
    })
)

results["predicted_label_name"] = (
    results["predicted_label"]
    .map({
        0: "Human",
        1: "AI"
    })
)

results["ai_probability"] = probabilities

results["human_probability"] = (
    1.0 - probabilities
)


OUTPUT_FILE = (
    r"datasets\text\HC3"
    r"\hc3_v2_validation_results.csv"
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("RESULTS SAVED")
print("=============")

print(
    "Output:",
    OUTPUT_FILE
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("V2 HC3 FINAL SUMMARY")
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
    "V2 HC3 HELD-OUT "
    "EVALUATION COMPLETED"
)