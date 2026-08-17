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
# V2.1 RAID EXTERNAL EVALUATION
# =========================================================

print()
print("=" * 60)
print("V2.1 MODEL — RAID EXTERNAL TEST")
print("=" * 60)


# =========================================================
# SETTINGS
# =========================================================

MODEL_DIR = r"models\text_final_v21"

TEST_FILE = (
    r"datasets\text\RAID"
    r"\raid_external.csv"
)

RESULT_FILE = (
    r"datasets\text\RAID"
    r"\raid_v21_final_results.csv"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_dataframe(df):

    df = df.copy()

    if "text" in df.columns:

        text_column = "text"

    elif "generation" in df.columns:

        text_column = "generation"

    else:

        raise ValueError(
            "No text or generation column found."
        )

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


        # -------------------------------------------------
        # Sentence count
        # -------------------------------------------------

        sentence_count = sum(
            1
            for char in text
            if char in ".!?"
        )

        if sentence_count == 0:

            sentence_count = 1


        # -------------------------------------------------
        # Average sentence length
        # -------------------------------------------------

        avg_sentence_length = (
            word_count /
            max(sentence_count, 1)
        )


        # -------------------------------------------------
        # Vocabulary diversity
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Log word count
        # -------------------------------------------------

        log_word_count = np.log1p(
            word_count
        )


        # -------------------------------------------------
        # Length bucket
        # -------------------------------------------------

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


def load_pickle(path):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(f)


# =========================================================
# LOAD RAID DATA
# =========================================================

print()
print("LOADING RAID EXTERNAL DATA")
print("==========================")


if not os.path.exists(TEST_FILE):

    raise FileNotFoundError(
        f"RAID external file not found:\n{TEST_FILE}"
    )


raid = pd.read_csv(
    TEST_FILE
)


raid = clean_dataframe(
    raid
)


# =========================================================
# CREATE LABEL
# =========================================================

if "label" in raid.columns:

    if raid["label"].dtype == object:

        raid["label"] = (
            raid["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "human": 0,
                    "ai": 1,
                    "0": 0,
                    "1": 1
                }
            )
        )

    else:

        raid["label"] = (
            raid["label"]
            .astype(int)
        )

elif "label_name" in raid.columns:

    raid["label"] = (
        raid["label_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "human": 0,
                "ai": 1
            }
        )
    )

else:

    raise ValueError(
        "RAID data does not contain "
        "'label' or 'label_name'."
    )


raid = raid[
    raid["label"].isin(
        [0, 1]
    )
].copy()


raid = add_length_group(
    raid
)


raid = raid.reset_index(
    drop=True
)


print(
    "External samples:",
    len(raid)
)


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    raid["label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
    .value_counts()
    .to_string()
)


# =========================================================
# MODEL DISTRIBUTION
# =========================================================

if "raid_model" in raid.columns:

    print()
    print("MODEL DISTRIBUTION")
    print("==================")

    print(
        raid["raid_model"]
        .value_counts()
        .to_string()
    )

elif "model" in raid.columns:

    print()
    print("MODEL DISTRIBUTION")
    print("==================")

    print(
        raid["model"]
        .value_counts()
        .to_string()
    )


# =========================================================
# DOMAIN DISTRIBUTION
# =========================================================

if "domain" in raid.columns:

    print()
    print("DOMAIN DISTRIBUTION")
    print("===================")

    print(
        raid["domain"]
        .value_counts()
        .to_string()
    )


# =========================================================
# TEXT LENGTH DISTRIBUTION
# =========================================================

print()
print("TEXT LENGTH DISTRIBUTION")
print("========================")


length_distribution = pd.crosstab(
    raid["length_group"],
    raid["label"]
)


length_distribution = (
    length_distribution
    .rename(
        columns={
            0: "Human",
            1: "AI"
        }
    )
)


print(
    length_distribution.to_string()
)


# =========================================================
# LOAD V2.1 MODEL
# =========================================================

print()
print("LOADING V2.1 MODEL")
print("==================")


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
    "V2.1 model components loaded."
)


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = (
    word_vectorizer.transform(
        raid["text"]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# =========================================================
# CHARACTER FEATURES
# =========================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        raid["text"]
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# =========================================================
# MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")


TRANSFORMER_NAME = config.get(
    "transformer",
    "sentence-transformers/all-MiniLM-L6-v2"
)


print(
    "Transformer:",
    TRANSFORMER_NAME
)


embedder = SentenceTransformer(
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


X_embedding = embedder.encode(
    raid["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    X_embedding.shape
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_linguistic_raw = (
    build_linguistic_features(
        raid["text"].tolist()
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


# =========================================================
# SPARSE FEATURES
# =========================================================

X_embedding_sparse = csr_matrix(
    X_embedding
)


X_linguistic_sparse = csr_matrix(
    X_linguistic
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("COMBINING FEATURES")
print("==================")


X_test = hstack(
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
    X_test.shape
)


# =========================================================
# FEATURE VALIDATION
# =========================================================

print()
print("FEATURE VALIDATION")
print("==================")


expected_features = config.get(
    "feature_count"
)


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
    X_test.shape[1]
)


if (
    expected_features is not None
    and
    X_test.shape[1] != expected_features
):

    raise RuntimeError(
        "Feature count does not match "
        "V2.1 configuration."
    )


if (
    X_test.shape[1]
    != classifier_features
):

    raise RuntimeError(
        "Feature count does not match "
        "classifier."
    )


print(
    "Feature configuration verified."
)


# =========================================================
# LABELS
# =========================================================

y_test = (
    raid["label"]
    .astype(int)
    .values
)


# =========================================================
# DECISION THRESHOLD
# =========================================================

threshold = float(
    config.get(
        "decision_threshold",
        0.50
    )
)


print()
print(
    "V2.1 decision threshold:",
    f"{threshold:.2f}"
)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("RUNNING V2.1 PREDICTIONS")
print("=========================")


probabilities = (
    classifier.predict_proba(
        X_test
    )[:, 1]
)


predictions = (
    probabilities
    >= threshold
).astype(int)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)


f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("V2.1 RAID EXTERNAL RESULTS")
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
        y_test,
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
        y_test,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# =========================================================
# PERFORMANCE BY GENERATOR
# =========================================================

model_column = None

if "raid_model" in raid.columns:

    model_column = "raid_model"

elif "model" in raid.columns:

    model_column = "model"


if model_column is not None:

    print()
    print("PERFORMANCE BY RAID GENERATOR")
    print("=============================")


    for generator in sorted(
        raid[model_column]
        .dropna()
        .unique()
    ):

        subset = raid[
            raid[model_column]
            == generator
        ].copy()


        indices = subset.index.values


        group_predictions = (
            predictions[indices]
        )


        group_labels = (
            subset["label"]
            .astype(int)
            .values
        )


        group_accuracy = (
            accuracy_score(
                group_labels,
                group_predictions
            )
        )


        group_f1 = (
            f1_score(
                group_labels,
                group_predictions,
                zero_division=0
            )
        )


        print(
            f"{str(generator):12s} "
            f"Accuracy={group_accuracy:.4f} "
            f"F1={group_f1:.4f} "
            f"Samples={len(subset)}"
        )


    # -----------------------------------------------------
    # AI RECALL BY GENERATOR
    # -----------------------------------------------------

    print()
    print("AI DETECTION BY GENERATOR")
    print("=========================")


    for generator in sorted(
        raid[model_column]
        .dropna()
        .unique()
    ):

        subset = raid[
            (
                raid[model_column]
                == generator
            )
            &
            (
                raid["label"]
                == 1
            )
        ].copy()


        if len(subset) == 0:

            continue


        indices = subset.index.values


        group_predictions = (
            predictions[indices]
        )


        group_labels = (
            subset["label"]
            .astype(int)
            .values
        )


        ai_recall = recall_score(
            group_labels,
            group_predictions,
            zero_division=0
        )


        print(
            f"{str(generator):12s} "
            f"AI Recall={ai_recall:.4f} "
            f"AI Samples={len(subset)}"
        )


# =========================================================
# PERFORMANCE BY DOMAIN
# =========================================================

if "domain" in raid.columns:

    print()
    print("PERFORMANCE BY DOMAIN")
    print("=====================")


    for domain in sorted(
        raid["domain"]
        .dropna()
        .unique()
    ):

        subset = raid[
            raid["domain"]
            == domain
        ].copy()


        indices = subset.index.values


        group_predictions = (
            predictions[indices]
        )


        group_labels = (
            subset["label"]
            .astype(int)
            .values
        )


        group_accuracy = (
            accuracy_score(
                group_labels,
                group_predictions
            )
        )


        group_f1 = (
            f1_score(
                group_labels,
                group_predictions,
                zero_division=0
            )
        )


        print(
            f"{str(domain):12s} "
            f"Accuracy={group_accuracy:.4f} "
            f"F1={group_f1:.4f} "
            f"Samples={len(subset)}"
        )


# =========================================================
# PERFORMANCE BY LENGTH
# =========================================================

print()
print("PERFORMANCE BY TEXT LENGTH")
print("===========================")


result_df = raid.copy()


result_df["prediction"] = (
    predictions
)


result_df["ai_probability"] = (
    probabilities
)


result_df["predicted_label"] = (
    result_df["prediction"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


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


    group_accuracy = (
        accuracy_score(
            subset["label"],
            subset["prediction"]
        )
    )


    group_precision = (
        precision_score(
            subset["label"],
            subset["prediction"],
            zero_division=0
        )
    )


    group_recall = (
        recall_score(
            subset["label"],
            subset["prediction"],
            zero_division=0
        )
    )


    group_f1 = (
        f1_score(
            subset["label"],
            subset["prediction"],
            zero_division=0
        )
    )


    print(
        f"{group:12s} "
        f"Accuracy={group_accuracy:.4f} "
        f"Precision={group_precision:.4f} "
        f"Recall={group_recall:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={len(subset)}"
    )


# =========================================================
# AI DETECTION BY LENGTH
# =========================================================

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


    ai_recall = recall_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    print(
        f"{group:12s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={len(subset)}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

print()
print("DETAILED RESULTS SAVED")
print("======================")


os.makedirs(
    os.path.dirname(
        RESULT_FILE
    ),
    exist_ok=True
)


result_df.to_csv(
    RESULT_FILE,
    index=False
)


print(
    "Output:",
    RESULT_FILE
)


# =========================================================
# FINAL SUMMARY
# =========================================================

correct = int(
    np.sum(
        predictions == y_test
    )
)


incorrect = (
    len(y_test)
    -
    correct
)


print()
print("=" * 60)
print("V2.1 RAID FINAL SUMMARY")
print("=" * 60)


print(
    f"Total samples        : {len(y_test)}"
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


print(
    f"ROC-AUC              : {roc_auc:.4f}"
)


print()
print(
    "V2.1 RAID EXTERNAL "
    "EVALUATION COMPLETED"
)