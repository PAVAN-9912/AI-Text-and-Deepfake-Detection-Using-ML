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
# CONFIGURATION
# =========================================================

MODEL_DIR = r"models\text_final_v2"

TEST_FILE = r"datasets\text\realworld_test.csv"

OUTPUT_FILE = (
    r"datasets\text\realworld_v2_results.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

THRESHOLD = 0.50


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("V2 MODEL — REAL-WORLD GENERALIZATION TEST")
print("=" * 60)
print()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

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

df = df[
    df["text"].str.len() > 0
].copy()

print(
    f"Test samples: {len(df)}"
)


# =========================================================
# LABELS
# =========================================================

y_true = (
    df["label"]
    .astype(int)
    .to_numpy()
)


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label"]
    .map({
        0: "Human",
        1: "AI"
    })
    .value_counts()
    .to_string()
)


# =========================================================
# CATEGORY DISTRIBUTION
# =========================================================

print()
print("CATEGORY DISTRIBUTION")
print("=====================")

print(
    df["category"]
    .value_counts()
    .to_string()
)


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("LENGTH DISTRIBUTION")
print("===================")

print(
    pd.crosstab(
        df["length_group"],
        df["label"].map({
            0: "Human",
            1: "AI"
        })
    )
    .to_string()
)


# =========================================================
# LOAD MODEL
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


print(
    "V2 model components loaded."
)


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
# PREDICTION
# =========================================================

print()
print("RUNNING V2 PREDICTIONS")
print("======================")

ai_probabilities = (
    classifier.predict_proba(
        combined_features
    )[:, 1]
)

predictions = (
    ai_probabilities >= THRESHOLD
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


print()
print("=" * 60)
print("REAL-WORLD V2 RESULTS")
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
# PERFORMANCE BY CATEGORY
# =========================================================

print()
print("PERFORMANCE BY CATEGORY")
print("=======================")

for category in df["category"].unique():

    mask = (
        df["category"]
        .to_numpy()
        ==
        category
    )

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
        f"{category:20s} "
        f"Accuracy={group_accuracy:.4f} "
        f"Precision={group_precision:.4f} "
        f"Recall={group_recall:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={mask.sum()}"
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
# INDIVIDUAL PREDICTIONS
# =========================================================

print()
print("=" * 60)
print("INDIVIDUAL PREDICTIONS")
print("=" * 60)

for index, row in df.iterrows():

    position = df.index.get_loc(index)

    actual = y_true[position]

    predicted = predictions[position]

    ai_probability = (
        ai_probabilities[position]
    )

    human_probability = (
        1.0 - ai_probability
    )

    actual_name = (
        "AI"
        if actual == 1
        else "Human"
    )

    predicted_name = (
        "AI"
        if predicted == 1
        else "Human"
    )

    status = (
        "CORRECT"
        if actual == predicted
        else "WRONG"
    )

    print()
    print("-" * 60)

    print(
        f"Category : {row['category']}"
    )

    print(
        f"Length   : {row['length_group']}"
    )

    print(
        f"Actual   : {actual_name}"
    )

    print(
        f"Predicted: {predicted_name}"
    )

    print(
        f"AI Prob. : {ai_probability * 100:.2f}%"
    )

    print(
        f"Human Prob.: {human_probability * 100:.2f}%"
    )

    print(
        f"Status   : {status}"
    )

    print(
        "Text:"
    )

    print(
        row["text"]
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results = df.copy()

results["actual_label"] = y_true

results["predicted_label"] = predictions

results["actual_label_name"] = (
    results["actual_label"]
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

results["ai_probability"] = (
    ai_probabilities
)

results["human_probability"] = (
    1.0 - ai_probabilities
)

results["correct"] = (
    results["actual_label"]
    ==
    results["predicted_label"]
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

correct = (
    y_true == predictions
).sum()

incorrect = (
    y_true != predictions
).sum()


print()
print("=" * 60)
print("REAL-WORLD V2 FINAL SUMMARY")
print("=" * 60)

print(
    f"Total samples        : {len(y_true)}"
)

print(
    f"Correct predictions  : {correct}"
)

print(
    f"Incorrect predictions: {incorrect}"
)

print(
    f"Accuracy             : {accuracy:.4f}"
)

print(
    f"F1 Score             : {f1:.4f}"
)

print()
print("RESULTS SAVED")
print("=============")

print(
    "Output:",
    OUTPUT_FILE
)

print()
print("=" * 60)
print("REAL-WORLD V2 EVALUATION COMPLETED")
print("=" * 60)