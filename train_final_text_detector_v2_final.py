import os
import pickle
import random

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


# =========================================================
# GENERALIZED FINAL TEXT DETECTOR V2
# =========================================================

print()
print("=" * 60)
print("GENERALIZED AI TEXT DETECTOR V2")
print("=" * 60)
print("HC3 + RAID + SentenceAI")
print("LENGTH-AWARE TRAINING")
print()


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


HC3_TRAIN = (
    r"datasets\text\HC3\train.csv"
)

HC3_VALIDATION = (
    r"datasets\text\HC3\validation.csv"
)

RAID_TRAIN = (
    r"datasets\text\RAID"
    r"\raid_domain_training_sample.csv"
)

SENTENCE_TRAIN = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_training.csv"
)


MODEL_DIR = (
    r"models\text_final_v2"
)


# =========================================================
# TF-IDF SETTINGS
# =========================================================

WORD_MAX_FEATURES = 150000
CHAR_MAX_FEATURES = 150000


# =========================================================
# TARGET TRAINING DISTRIBUTION
# =========================================================
#
# We deliberately balance Human/AI AND length.
#
# The goal is not to throw away everything.
# We use controlled sampling so that long text does
# not dominate the classifier.
#
# Target is per LABEL per LENGTH GROUP.
#
# These targets are chosen conservatively based on
# the available datasets.
# =========================================================

TARGETS = {

    "very_short": 1500,

    "short": 2500,

    "medium": 1500,

    "long": 500
}


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

        words = str(text).split()

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


def save_pickle(obj, path):

    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(
            obj,
            f
        )


# =========================================================
# BALANCED SAMPLING FUNCTION
# =========================================================

def sample_length_label(
    df,
    label,
    length_group,
    target,
    dataset_name
):

    subset = df[
        (df["label"] == label)
        &
        (df["length_group"] == length_group)
    ].copy()


    available = len(subset)


    print(
        f"{dataset_name:12s} "
        f"{length_group:11s} "
        f"{'Human' if label == 0 else 'AI':6s} "
        f"Available={available:5d} "
        f"Target={target:5d}"
    )


    if available == 0:

        return pd.DataFrame()


    # If there are more samples than needed,
    # sample randomly.
    if available >= target:

        return subset.sample(
            n=target,
            random_state=RANDOM_SEED
        )


    # If there are fewer samples, keep all.
    #
    # We do NOT duplicate data artificially.
    return subset


# =========================================================
# LOAD HC3
# =========================================================

print()
print("LOADING HC3")
print("===========")


hc3_train = pd.read_csv(
    HC3_TRAIN
)

hc3_validation = pd.read_csv(
    HC3_VALIDATION
)


hc3_train = clean_dataframe(
    hc3_train
)

hc3_validation = clean_dataframe(
    hc3_validation
)


hc3_train = add_length_group(
    hc3_train
)

hc3_validation = add_length_group(
    hc3_validation
)


print(
    "HC3 training samples:",
    len(hc3_train)
)

print(
    "HC3 validation samples:",
    len(hc3_validation)
)


# =========================================================
# LOAD RAID
# =========================================================

print()
print("LOADING RAID")
print("============")


raid = pd.read_csv(
    RAID_TRAIN
)

raid = clean_dataframe(
    raid
)

raid = add_length_group(
    raid
)


print(
    "RAID samples:",
    len(raid)
)


# =========================================================
# LOAD SENTENCEAI
# =========================================================

print()
print("LOADING SENTENCEAI")
print("==================")


sentence_train = pd.read_csv(
    SENTENCE_TRAIN
)

sentence_train = clean_dataframe(
    sentence_train
)


# SentenceAI uses label_name rather than label.
sentence_train["label"] = (
    sentence_train["label_name"]
    .map(
        {
            "Human": 0,
            "AI": 1
        }
    )
)


sentence_train = sentence_train[
    sentence_train["label"].isin(
        [0, 1]
    )
].copy()


sentence_train = add_length_group(
    sentence_train
)


print(
    "SentenceAI samples:",
    len(sentence_train)
)


# =========================================================
# SHOW ORIGINAL DISTRIBUTION
# =========================================================

print()
print("ORIGINAL LENGTH DISTRIBUTION")
print("============================")


for name, df in [
    ("HC3", hc3_train),
    ("RAID", raid),
    ("SentenceAI", sentence_train)
]:

    print()
    print(name)

    print(
        pd.crosstab(
            df["length_group"],
            df["label"]
        )
        .to_string()
    )


# =========================================================
# BUILD LENGTH-AWARE DATASET
# =========================================================

print()
print("BUILDING LENGTH-AWARE TRAINING DATA")
print("===================================")


# ---------------------------------------------------------
# First priority:
#
# SentenceAI is specifically valuable for short text.
# We use it heavily for very_short and short samples.
# ---------------------------------------------------------

selected_parts = []


for length_group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    target = TARGETS[
        length_group
    ]


    print()
    print(
        f"PROCESSING {length_group.upper()}"
    )

    print(
        "-" * 45
    )


    # =====================================================
    # SENTENCEAI
    # =====================================================

    sentence_target = 0

    if length_group == "very_short":

        sentence_target = min(
            1000,
            target
        )

    elif length_group == "short":

        sentence_target = min(
            1500,
            target
        )

    elif length_group == "medium":

        sentence_target = min(
            500,
            target
        )


    if sentence_target > 0:

        for label in [0, 1]:

            sampled = sample_length_label(
                sentence_train,
                label,
                length_group,
                sentence_target,
                "SentenceAI"
            )

            if len(sampled) > 0:

                sampled = sampled.copy()

                sampled["dataset"] = (
                    "SentenceAI"
                )

                selected_parts.append(
                    sampled
                )


    # =====================================================
    # HC3
    # =====================================================

    # Remaining target after SentenceAI.
    remaining_target = (
        target -
        sentence_target
    )


    # Split remaining target equally
    # between HC3 and RAID when possible.
    hc3_target = (
        remaining_target // 2
    )

    raid_target = (
        remaining_target -
        hc3_target
    )


    # -----------------------------------------------------
    # HC3
    # -----------------------------------------------------

    for label in [0, 1]:

        sampled = sample_length_label(
            hc3_train,
            label,
            length_group,
            hc3_target,
            "HC3"
        )

        if len(sampled) > 0:

            sampled = sampled.copy()

            sampled["dataset"] = "HC3"

            selected_parts.append(
                sampled
            )


    # -----------------------------------------------------
    # RAID
    # -----------------------------------------------------

    for label in [0, 1]:

        subset = raid[
            (raid["label"] == label)
            &
            (
                raid["length_group"]
                == length_group
            )
        ].copy()


        # For RAID AI, preserve generator diversity.
        if (
            label == 1
            and
            len(subset) > 0
            and
            "raid_model" in subset.columns
        ):

            generators = [
                "gpt2",
                "llama-chat",
                "mpt",
                "mpt-chat"
            ]


            per_generator = max(
                1,
                raid_target //
                len(generators)
            )


            generator_parts = []


            for generator in generators:

                generator_df = subset[
                    subset["raid_model"]
                    == generator
                ]

                if len(generator_df) == 0:
                    continue


                take = min(
                    per_generator,
                    len(generator_df)
                )


                generator_parts.append(
                    generator_df.sample(
                        n=take,
                        random_state=RANDOM_SEED
                    )
                )


            if generator_parts:

                sampled = pd.concat(
                    generator_parts,
                    ignore_index=True
                )


                # Fill remaining slots.
                if len(sampled) < raid_target:

                    used_indices = set(
                        sampled.index
                    )

                    remaining_pool = subset[
                        ~subset.index.isin(
                            used_indices
                        )
                    ]


                    extra_needed = (
                        raid_target -
                        len(sampled)
                    )


                    if (
                        extra_needed > 0
                        and
                        len(remaining_pool) > 0
                    ):

                        extra = remaining_pool.sample(
                            n=min(
                                extra_needed,
                                len(remaining_pool)
                            ),
                            random_state=RANDOM_SEED
                        )

                        sampled = pd.concat(
                            [
                                sampled,
                                extra
                            ],
                            ignore_index=True
                        )

            else:

                sampled = subset.sample(
                    n=min(
                        raid_target,
                        len(subset)
                    ),
                    random_state=RANDOM_SEED
                )

        else:

            sampled = subset.sample(
                n=min(
                    raid_target,
                    len(subset)
                ),
                random_state=RANDOM_SEED
            ) if len(subset) > 0 else pd.DataFrame()


        if len(sampled) > 0:

            sampled = sampled.copy()

            sampled["dataset"] = "RAID"

            selected_parts.append(
                sampled
            )


# =========================================================
# COMBINE
# =========================================================

train_df = pd.concat(
    selected_parts,
    ignore_index=True
)


# Remove accidental duplicates.
train_df = train_df.drop_duplicates(
    subset=["text"]
).reset_index(
    drop=True
)


# Shuffle.
train_df = train_df.sample(
    frac=1.0,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


# =========================================================
# FINAL DISTRIBUTION
# =========================================================

print()
print("FINAL TRAINING DISTRIBUTION")
print("===========================")


print()
print(
    "Total samples:",
    len(train_df)
)


print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    train_df[
        "label"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("DATASET DISTRIBUTION")
print("====================")


print(
    pd.crosstab(
        train_df["dataset"],
        train_df["label"]
    ).to_string()
)


print()
print("LENGTH DISTRIBUTION")
print("===================")


print(
    pd.crosstab(
        train_df["length_group"],
        train_df["label"]
    ).to_string()
)


# =========================================================
# CREATE A MIXED VALIDATION SET
# =========================================================
#
# We do NOT use external SentenceAI test or RAID external
# data here.
#
# We hold out samples from the training sources instead.
# =========================================================

print()
print("BUILDING INTERNAL MIXED VALIDATION SET")
print("======================================")


validation_parts = []


for dataset_name in [
    "HC3",
    "RAID",
    "SentenceAI"
]:

    dataset_df = train_df[
        train_df["dataset"]
        == dataset_name
    ].copy()


    if len(dataset_df) == 0:
        continue


    # -----------------------------------------------------
    # Build the validation subset explicitly instead of
    # using groupby().apply().
    #
    # This avoids pandas-version differences where
    # groupby().apply() can drop or move grouping columns.
    # Every sampled row keeps its original label and
    # length_group columns.
    # -----------------------------------------------------

    dataset_validation_parts = []

    for label_value in sorted(
        dataset_df["label"].unique()
    ):

        for length_value in sorted(
            dataset_df["length_group"].unique()
        ):

            group_df = dataset_df[
                (dataset_df["label"] == label_value)
                &
                (dataset_df["length_group"] == length_value)
            ].copy()

            if len(group_df) == 0:
                continue

            take = max(
                1,
                int(len(group_df) * 0.10)
            )

            take = min(
                take,
                len(group_df)
            )

            sampled = group_df.sample(
                n=take,
                random_state=RANDOM_SEED
            ).copy()

            dataset_validation_parts.append(
                sampled
            )

    if dataset_validation_parts:

        validation_part = pd.concat(
            dataset_validation_parts,
            ignore_index=True
        )

        validation_parts.append(
            validation_part
        )


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)

# ---------------------------------------------------------
# VALIDATION COLUMN NORMALIZATION
# ---------------------------------------------------------
# Validation rows were sampled explicitly above, so the
# label column is guaranteed to remain a normal DataFrame
# column. Recompute length_group from text as a final guard.
# ---------------------------------------------------------

if "label" not in validation_df.columns:
    raise RuntimeError(
        "Validation dataframe does not contain the label column."
    )

validation_df["label"] = (
    validation_df["label"]
    .astype(int)
)

validation_df["length_group"] = (
    validation_df["text"]
    .apply(get_length_group)
)

validation_df = validation_df.reset_index(
    drop=True
)


# Remove validation texts from training.
validation_texts = set(
    validation_df["text"]
)


train_df = train_df[
    ~train_df["text"].isin(
        validation_texts
    )
].copy()


train_df = train_df.reset_index(
    drop=True
)

validation_df = validation_df.reset_index(
    drop=True
)


print()
print(
    "Training samples after split:",
    len(train_df)
)

print(
    "Mixed validation samples:",
    len(validation_df)
)


print()
print("VALIDATION LENGTH DISTRIBUTION")
print("==============================")


print(
    pd.crosstab(
        validation_df["length_group"],
        validation_df["label"]
    ).to_string()
)


# =========================================================
# WORD TF-IDF
# =========================================================

print()
print("BUILDING WORD TF-IDF")
print("====================")


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=WORD_MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    max_df=0.98
)


print(
    "Fitting word vocabulary..."
)


X_train_word = (
    word_vectorizer.fit_transform(
        train_df["text"]
    )
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


# =========================================================
# CHARACTER TF-IDF
# =========================================================

print()
print("BUILDING CHARACTER TF-IDF")
print("=========================")


char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    max_df=0.99
)


print(
    "Fitting character vocabulary..."
)


X_train_char = (
    char_vectorizer.fit_transform(
        train_df["text"]
    )
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


embedder = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# =========================================================
# TRAINING EMBEDDINGS
# =========================================================

print()
print("CREATING TRAINING EMBEDDINGS")
print("============================")


X_train_embedding = embedder.encode(
    train_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Training embedding shape:",
    X_train_embedding.shape
)


# =========================================================
# VALIDATION EMBEDDINGS
# =========================================================

print()
print("CREATING VALIDATION EMBEDDINGS")
print("==============================")


X_validation_embedding = embedder.encode(
    validation_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Validation embedding shape:",
    X_validation_embedding.shape
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_train_linguistic_raw = (
    build_linguistic_features(
        train_df["text"].tolist()
    )
)


X_validation_linguistic_raw = (
    build_linguistic_features(
        validation_df["text"].tolist()
    )
)


print(
    "Training linguistic shape:",
    X_train_linguistic_raw.shape
)

print(
    "Validation linguistic shape:",
    X_validation_linguistic_raw.shape
)


# =========================================================
# SCALE
# =========================================================

print()
print("SCALING LINGUISTIC FEATURES")
print("===========================")


linguistic_scaler = StandardScaler()


X_train_linguistic = (
    linguistic_scaler.fit_transform(
        X_train_linguistic_raw
    )
)


X_validation_linguistic = (
    linguistic_scaler.transform(
        X_validation_linguistic_raw
    )
)


# =========================================================
# SPARSE CONVERSION
# =========================================================

X_train_embedding_sparse = csr_matrix(
    X_train_embedding
)

X_validation_embedding_sparse = csr_matrix(
    X_validation_embedding
)


X_train_linguistic_sparse = csr_matrix(
    X_train_linguistic
)

X_validation_linguistic_sparse = csr_matrix(
    X_validation_linguistic
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("COMBINING FEATURES")
print("==================")


X_train = hstack(
    [
        X_train_word,
        X_train_char,
        X_train_embedding_sparse,
        X_train_linguistic_sparse
    ],
    format="csr"
)


X_validation = hstack(
    [
        X_validation_word,
        X_validation_char,
        X_validation_embedding_sparse,
        X_validation_linguistic_sparse
    ],
    format="csr"
)


print(
    "Combined training shape:",
    X_train.shape
)

print(
    "Combined validation shape:",
    X_validation.shape
)


# =========================================================
# LABELS
# =========================================================

y_train = (
    train_df["label"]
    .astype(int)
    .values
)

y_validation = (
    validation_df["label"]
    .astype(int)
    .values
)


# =========================================================
# TRAINING SAMPLE WEIGHTS
# =========================================================
#
# Since the data is already much better balanced, we use
# only a mild weighting.
#
# This prevents the classifier from becoming excessively
# biased toward long text.
# =========================================================

print()
print("BUILDING SAMPLE WEIGHTS")
print("=======================")


length_weights = {

    "very_short": 1.25,

    "short": 1.15,

    "medium": 1.05,

    "long": 1.00
}


sample_weights = (
    train_df["length_group"]
    .map(length_weights)
    .astype(float)
    .values
)


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print()
print("TRAINING V2 FUSION CLASSIFIER")
print("=============================")


classifier = LogisticRegression(
    max_iter=2500,
    C=2.0,
    solver="liblinear",
    class_weight="balanced",
    random_state=RANDOM_SEED
)


classifier.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


print(
    "Training completed."
)


# =========================================================
# VALIDATION PROBABILITIES
# =========================================================

validation_probabilities = (
    classifier.predict_proba(
        X_validation
    )[:, 1]
)


# =========================================================
# THRESHOLD SEARCH
# =========================================================
#
# We choose the threshold using validation F1.
#
# This avoids blindly assuming 0.50 is optimal.
# =========================================================

print()
print("CALIBRATING DECISION THRESHOLD")
print("===============================")


best_threshold = 0.50
best_f1 = -1.0


threshold_results = []


for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


    current_f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )


    current_accuracy = accuracy_score(
        y_validation,
        predictions
    )


    threshold_results.append(
        {
            "threshold":
                float(threshold),

            "f1":
                float(current_f1),

            "accuracy":
                float(current_accuracy)
        }
    )


    if current_f1 > best_f1:

        best_f1 = current_f1

        best_threshold = (
            float(threshold)
        )


print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1 at threshold: "
    f"{best_f1:.4f}"
)


# =========================================================
# FINAL VALIDATION PREDICTIONS
# =========================================================

validation_predictions = (
    validation_probabilities
    >= best_threshold
).astype(int)


# =========================================================
# METRICS
# =========================================================

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


roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("V2 VALIDATION RESULTS")
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
    f"Threshold: {best_threshold:.2f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print("================")


print(
    confusion_matrix(
        y_validation,
        validation_predictions
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
        y_validation,
        validation_predictions,
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
print("VALIDATION PERFORMANCE BY LENGTH")
print("================================")


validation_result_df = (
    validation_df.copy()
)


validation_result_df[
    "prediction"
] = validation_predictions


validation_result_df[
    "probability"
] = validation_probabilities


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = validation_result_df[
        validation_result_df[
            "length_group"
        ]
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


# =========================================================
# SAVE MODEL
# =========================================================

print()
print("SAVING V2 MODEL")
print("===============")


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


save_pickle(
    word_vectorizer,
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)


save_pickle(
    char_vectorizer,
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)


save_pickle(
    linguistic_scaler,
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)


save_pickle(
    classifier,
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


# =========================================================
# CONFIG
# =========================================================

config = {

    "transformer":
        TRANSFORMER_NAME,

    "word_max_features":
        WORD_MAX_FEATURES,

    "char_max_features":
        CHAR_MAX_FEATURES,

    "feature_count":
        X_train.shape[1],

    "linguistic_features": [

        "log_word_count",

        "sentence_count",

        "avg_sentence_length",

        "vocabulary_diversity",

        "length_bucket"
    ],

    "random_seed":
        RANDOM_SEED,

    "training_samples":
        len(train_df),

    "validation_samples":
        len(validation_df),

    "training_datasets": [

        "HC3",

        "RAID",

        "SentenceAI"
    ],

    "decision_threshold":
        best_threshold,

    "length_targets":
        TARGETS,

    "model_version":
        "text_final_v2"
}


save_pickle(
    config,
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


# =========================================================
# METRICS
# =========================================================

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
        float(roc_auc),

    "decision_threshold":
        float(best_threshold),

    "training_samples":
        len(train_df),

    "validation_samples":
        len(validation_df)
}


save_pickle(
    metrics,
    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    )
)


# =========================================================
# SAVE TRAINING DISTRIBUTION
# =========================================================

distribution = (
    pd.crosstab(
        [
            train_df["dataset"],
            train_df["length_group"]
        ],
        train_df["label"]
    )
    .reset_index()
)


distribution.to_csv(
    os.path.join(
        MODEL_DIR,
        "training_distribution.csv"
    ),
    index=False
)


# =========================================================
# SAVE VALIDATION RESULTS
# =========================================================

validation_result_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "validation_results.csv"
    ),
    index=False
)


# =========================================================
# SAVE THRESHOLD RESULTS
# =========================================================

pd.DataFrame(
    threshold_results
).to_csv(
    os.path.join(
        MODEL_DIR,
        "threshold_results.csv"
    ),
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 60)
print("V2 MODEL SAVED")
print("=" * 60)


print(
    "Directory:",
    MODEL_DIR
)


print()
print("Files:")


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

print(
    " - training_distribution.csv"
)

print(
    " - validation_results.csv"
)

print(
    " - threshold_results.csv"
)


print()
print("=" * 60)
print("GENERALIZED TEXT DETECTOR V2 TRAINING COMPLETED")
print("=" * 60)