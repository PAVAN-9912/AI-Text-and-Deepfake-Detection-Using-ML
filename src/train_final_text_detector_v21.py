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
# GENERALIZED FINAL TEXT DETECTOR V2.1
# =========================================================

print()
print("=" * 60)
print("GENERALIZED AI TEXT DETECTOR V2.1")
print("=" * 60)
print("HC3 + RAID + SentenceAI")
print("SOURCE + LENGTH BALANCED TRAINING")
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


# IMPORTANT:
# V2 is NOT overwritten.

MODEL_DIR = (
    r"models\text_final_v21"
)


# =========================================================
# TF-IDF SETTINGS
# =========================================================

WORD_MAX_FEATURES = 150000

CHAR_MAX_FEATURES = 150000


# =========================================================
# TARGET DISTRIBUTION
# =========================================================
#
# Target is PER LABEL per LENGTH GROUP.
#
# This is intentionally smaller than simply taking huge
# amounts of HC3 long text.
#
# Goal:
#
# Human / AI balanced
# Length balanced
# Dataset diversity preserved
#
# Total target:
#
# 1000 + 1800 + 1200 + 600 = 4600
# per label
#
# Approximately 9200 samples before validation.
# =========================================================

TARGETS = {

    "very_short": 1000,

    "short": 1800,

    "medium": 1200,

    "long": 600
}


LENGTH_GROUPS = [
    "very_short",
    "short",
    "medium",
    "long"
]


DATASETS = [
    "HC3",
    "RAID",
    "SentenceAI"
]


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
# SOURCE-BALANCED SAMPLING
# =========================================================
#
# Instead of:
#
# SentenceAI first
# then HC3 + RAID get leftovers
#
# V2.1 pools all three datasets for each
# label + length combination.
#
# We try to obtain samples from every available source.
#
# This reduces the chance that the model learns:
#
# "short AI = SentenceAI"
#
# or
#
# "long text = HC3"
#
# =========================================================

def source_balanced_sample(
    dataset_frames,
    label,
    length_group,
    target
):

    available_frames = []

    print()
    print(
        f"PROCESSING {length_group.upper()} "
        f"— {'HUMAN' if label == 0 else 'AI'}"
    )

    print(
        "-" * 55
    )


    # -----------------------------------------------------
    # Build source subsets
    # -----------------------------------------------------

    for dataset_name in DATASETS:

        df = dataset_frames[
            dataset_name
        ]

        subset = df[
            (df["label"] == label)
            &
            (
                df["length_group"]
                == length_group
            )
        ].copy()


        if len(subset) == 0:

            print(
                f"{dataset_name:12s} "
                f"Available=0"
            )

            continue


        available_frames.append(
            (
                dataset_name,
                subset
            )
        )


        print(
            f"{dataset_name:12s} "
            f"Available={len(subset):5d}"
        )


    if not available_frames:

        return pd.DataFrame()


    # -----------------------------------------------------
    # Special RAID AI handling
    #
    # Preserve generator diversity.
    # -----------------------------------------------------

    selected_parts = []

    remaining_target = target


    raid_entry = None

    for item in available_frames:

        if item[0] == "RAID":

            raid_entry = item

            break


    if raid_entry is not None and label == 1:

        raid_df = raid_entry[1]

        if (
            "raid_model" in
            raid_df.columns
        ):

            generators = [
                "gpt2",
                "llama-chat",
                "mpt",
                "mpt-chat"
            ]


            generator_target = max(
                1,
                target //
                len(generators)
            )


            raid_parts = []


            for generator in generators:

                generator_df = raid_df[
                    raid_df["raid_model"]
                    == generator
                ]


                if len(generator_df) == 0:
                    continue


                take = min(
                    generator_target,
                    len(generator_df)
                )


                if take > 0:

                    part = generator_df.sample(
                        n=take,
                        random_state=(
                            RANDOM_SEED
                            + len(
                                raid_parts
                            )
                        )
                    ).copy()


                    raid_parts.append(
                        part
                    )


            if raid_parts:

                raid_selected = pd.concat(
                    raid_parts,
                    ignore_index=True
                )


                selected_parts.append(
                    raid_selected
                )


                remaining_target -= (
                    len(raid_selected)
                )


                remaining_target = max(
                    0,
                    remaining_target
                )


    # -----------------------------------------------------
    # Remove RAID from normal allocation if RAID AI
    # was already sampled.
    # -----------------------------------------------------

    allocation_frames = []


    for dataset_name, subset in available_frames:

        if (
            dataset_name == "RAID"
            and label == 1
            and remaining_target < target
        ):

            # Exclude already selected RAID rows
            # using text to avoid accidental duplication.

            if selected_parts:

                used_texts = set(
                    pd.concat(
                        selected_parts,
                        ignore_index=True
                    )["text"]
                )

                subset = subset[
                    ~subset["text"].isin(
                        used_texts
                    )
                ].copy()


        if len(subset) > 0:

            allocation_frames.append(
                (
                    dataset_name,
                    subset
                )
            )


    # -----------------------------------------------------
    # Allocate remaining samples.
    #
    # First give each source a basic share.
    # Then distribute remaining slots according
    # to availability.
    # -----------------------------------------------------

    if remaining_target > 0:

        source_count = len(
            allocation_frames
        )


        if source_count > 0:

            base_share = (
                remaining_target //
                source_count
            )


            allocated = 0


            for dataset_name, subset in allocation_frames:

                take = min(
                    base_share,
                    len(subset)
                )


                if take > 0:

                    part = subset.sample(
                        n=take,
                        random_state=RANDOM_SEED
                    ).copy()


                    selected_parts.append(
                        part
                    )


                    allocated += take


            remaining_target -= allocated


            remaining_target = max(
                0,
                remaining_target
            )


        # -------------------------------------------------
        # Fill remaining slots proportionally.
        # -------------------------------------------------

        if remaining_target > 0:

            used_texts = set()

            if selected_parts:

                used_texts = set(
                    pd.concat(
                        selected_parts,
                        ignore_index=True
                    )["text"]
                )


            remaining_pools = []


            for dataset_name, subset in allocation_frames:

                pool = subset[
                    ~subset["text"].isin(
                        used_texts
                    )
                ].copy()


                if len(pool) > 0:

                    remaining_pools.append(
                        (
                            dataset_name,
                            pool
                        )
                    )


            # -------------------------------------------------
            # Iterative proportional filling.
            # -------------------------------------------------

            while (
                remaining_target > 0
                and
                remaining_pools
            ):

                total_available = sum(
                    len(pool)
                    for _, pool
                    in remaining_pools
                )


                if total_available == 0:
                    break


                new_pools = []


                progress = False


                for dataset_name, pool in remaining_pools:

                    if remaining_target <= 0:
                        break


                    proportional = int(
                        round(
                            remaining_target
                            *
                            len(pool)
                            /
                            total_available
                        )
                    )


                    take = min(
                        proportional,
                        len(pool),
                        remaining_target
                    )


                    if (
                        take == 0
                        and
                        len(pool) > 0
                    ):

                        take = 1


                    take = min(
                        take,
                        remaining_target
                    )


                    if take > 0:

                        part = pool.sample(
                            n=take,
                            random_state=(
                                RANDOM_SEED
                                +
                                remaining_target
                            )
                        ).copy()


                        selected_parts.append(
                            part
                        )


                        remaining_target -= take

                        progress = True


                        selected_texts = set(
                            part["text"]
                        )


                        pool = pool[
                            ~pool["text"].isin(
                                selected_texts
                            )
                        ].copy()


                    if len(pool) > 0:

                        new_pools.append(
                            (
                                dataset_name,
                                pool
                            )
                        )


                remaining_pools = new_pools


                if not progress:
                    break


    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    if not selected_parts:

        return pd.DataFrame()


    result = pd.concat(
        selected_parts,
        ignore_index=True
    )


    # Never exceed target.

    if len(result) > target:

        result = result.sample(
            n=target,
            random_state=RANDOM_SEED
        )


    return result.reset_index(
        drop=True
    )


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
# ORIGINAL DISTRIBUTION
# =========================================================

print()
print("=" * 60)
print("ORIGINAL DATA DISTRIBUTIONS")
print("=" * 60)


for name, df in [
    ("HC3", hc3_train),
    ("RAID", raid),
    ("SentenceAI", sentence_train)
]:

    print()
    print(name)
    print("=" * len(name))


    print()
    print(
        pd.crosstab(
            df["length_group"],
            df["label"]
        ).to_string()
    )


# =========================================================
# DATASET DICTIONARY
# =========================================================

dataset_frames = {

    "HC3":
        hc3_train,

    "RAID":
        raid,

    "SentenceAI":
        sentence_train
}


# =========================================================
# BUILD V2.1 TRAINING DATA
# =========================================================

print()
print("=" * 60)
print("BUILDING V2.1 SOURCE + LENGTH BALANCED DATA")
print("=" * 60)


selected_parts = []


for length_group in LENGTH_GROUPS:

    target = TARGETS[
        length_group
    ]


    for label in [0, 1]:

        sampled = source_balanced_sample(
            dataset_frames,
            label,
            length_group,
            target
        )


        if len(sampled) == 0:

            continue


        sampled = sampled.copy()


        if "dataset" not in sampled.columns:

            # Dataset should already exist for the
            # source-specific frames, but this is a guard.

            pass


        selected_parts.append(
            sampled
        )


# =========================================================
# COMBINE TRAINING DATA
# =========================================================

train_df = pd.concat(
    selected_parts,
    ignore_index=True
)


# ---------------------------------------------------------
# Normalize dataset labels.
# ---------------------------------------------------------

# Because each source dataframe may not have had the
# dataset column, identify it by original text membership
# if necessary.

if "dataset" not in train_df.columns:

    train_df["dataset"] = ""


# The sampling function returns rows from the original
# source data. We therefore rebuild dataset information
# deterministically using text membership.

hc3_texts = set(
    hc3_train["text"]
)

raid_texts = set(
    raid["text"]
)

sentence_texts = set(
    sentence_train["text"]
)


def identify_dataset(text):

    if text in hc3_texts:
        return "HC3"

    if text in raid_texts:
        return "RAID"

    if text in sentence_texts:
        return "SentenceAI"

    return "Unknown"


train_df["dataset"] = (
    train_df["text"]
    .apply(identify_dataset)
)


# ---------------------------------------------------------
# Remove accidental duplicates.
# ---------------------------------------------------------

train_df = train_df.drop_duplicates(
    subset=["text"]
).reset_index(
    drop=True
)


# ---------------------------------------------------------
# Recompute length group.
# ---------------------------------------------------------

train_df["length_group"] = (
    train_df["text"]
    .apply(get_length_group)
)


# ---------------------------------------------------------
# Shuffle.
# ---------------------------------------------------------

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
print("=" * 60)
print("V2.1 FINAL TRAINING DISTRIBUTION")
print("=" * 60)


print()
print(
    "Total samples:",
    len(train_df)
)


print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    train_df["label"]
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
# INTERNAL MIXED VALIDATION
# =========================================================

print()
print("=" * 60)
print("BUILDING V2.1 INTERNAL VALIDATION SET")
print("=" * 60)


validation_parts = []


for dataset_name in DATASETS:

    dataset_df = train_df[
        train_df["dataset"]
        == dataset_name
    ].copy()


    if len(dataset_df) == 0:
        continue


    dataset_validation_parts = []


    for label_value in sorted(
        dataset_df["label"].unique()
    ):

        for length_value in LENGTH_GROUPS:

            group_df = dataset_df[
                (dataset_df["label"] == label_value)
                &
                (
                    dataset_df["length_group"]
                    == length_value
                )
            ].copy()


            if len(group_df) == 0:
                continue


            take = max(
                1,
                int(
                    len(group_df) * 0.10
                )
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

        validation_parts.append(
            pd.concat(
                dataset_validation_parts,
                ignore_index=True
            )
        )


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
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


# ---------------------------------------------------------
# Remove validation texts from training.
# ---------------------------------------------------------

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


print()
print(
    "Training samples:",
    len(train_df)
)

print(
    "Validation samples:",
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
print("=" * 60)
print("BUILDING WORD TF-IDF")
print("=" * 60)


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
print("=" * 60)
print("BUILDING CHARACTER TF-IDF")
print("=" * 60)


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
print("=" * 60)
print("LOADING MINILM")
print("=" * 60)


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
print("=" * 60)
print("CREATING TRAINING EMBEDDINGS")
print("=" * 60)


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
print("=" * 60)
print("CREATING VALIDATION EMBEDDINGS")
print("=" * 60)


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
print("=" * 60)
print("BUILDING LINGUISTIC FEATURES")
print("=" * 60)


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
# SCALE LINGUISTIC FEATURES
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
print("=" * 60)
print("COMBINING FEATURES")
print("=" * 60)


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


expected_features = (
    X_train_word.shape[1]
    +
    X_train_char.shape[1]
    +
    X_train_embedding.shape[1]
    +
    X_train_linguistic.shape[1]
)


print()
print(
    "Expected feature count:",
    expected_features
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
# SAMPLE WEIGHTS
# =========================================================
#
# Slightly emphasize short text.
#
# This is intentionally mild.
# We do NOT make length itself determine the prediction.
# =========================================================

print()
print("=" * 60)
print("BUILDING SAMPLE WEIGHTS")
print("=" * 60)


length_weights = {

    "very_short": 1.30,

    "short": 1.20,

    "medium": 1.05,

    "long": 1.00
}


dataset_weights = {

    "HC3": 1.00,

    "RAID": 1.05,

    "SentenceAI": 1.00
}


length_weight_values = (
    train_df["length_group"]
    .map(length_weights)
    .astype(float)
)


dataset_weight_values = (
    train_df["dataset"]
    .map(dataset_weights)
    .fillna(1.0)
    .astype(float)
)


sample_weights = (
    length_weight_values
    *
    dataset_weight_values
).values


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print()
print("=" * 60)
print("TRAINING V2.1 FUSION CLASSIFIER")
print("=" * 60)


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
# BALANCED THRESHOLD SEARCH
# =========================================================
#
# IMPORTANT CHANGE FROM V2:
#
# Instead of optimizing only overall F1, we give equal
# importance to the four text-length groups.
#
# This prevents the huge long-text group from dominating
# threshold selection.
# =========================================================

print()
print("=" * 60)
print("CALIBRATING V2.1 DECISION THRESHOLD")
print("=" * 60)


threshold_results = []


best_threshold = 0.50
best_balanced_score = -1.0


for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


    overall_f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )


    length_f1_values = []


    for group in LENGTH_GROUPS:

        mask = (
            validation_df["length_group"]
            == group
        ).values


        if not np.any(mask):
            continue


        group_f1 = f1_score(
            y_validation[mask],
            predictions[mask],
            zero_division=0
        )


        length_f1_values.append(
            group_f1
        )


    if length_f1_values:

        balanced_length_f1 = float(
            np.mean(
                length_f1_values
            )
        )

    else:

        balanced_length_f1 = (
            overall_f1
        )


    # Give overall F1 some importance while still
    # protecting short-text performance.

    objective = (
        0.60 * overall_f1
        +
        0.40 * balanced_length_f1
    )


    current_accuracy = (
        accuracy_score(
            y_validation,
            predictions
        )
    )


    threshold_results.append(
        {
            "threshold":
                float(threshold),

            "overall_f1":
                float(overall_f1),

            "balanced_length_f1":
                float(
                    balanced_length_f1
                ),

            "objective":
                float(objective),

            "accuracy":
                float(
                    current_accuracy
                )
        }
    )


    if objective > best_balanced_score:

        best_balanced_score = objective

        best_threshold = (
            float(threshold)
        )


print()
print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Balanced validation objective: "
    f"{best_balanced_score:.4f}"
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
print("V2.1 VALIDATION RESULTS")
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
print("=" * 60)
print("V2.1 PERFORMANCE BY TEXT LENGTH")
print("=" * 60)


validation_result_df = (
    validation_df.copy()
)


validation_result_df[
    "prediction"
] = validation_predictions


validation_result_df[
    "probability"
] = validation_probabilities


length_metrics = []


for group in LENGTH_GROUPS:

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


    length_metrics.append(
        {
            "length_group": group,

            "accuracy":
                float(group_accuracy),

            "precision":
                float(group_precision),

            "recall":
                float(group_recall),

            "f1":
                float(group_f1),

            "samples":
                int(len(subset))
        }
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
print("=" * 60)
print("SAVING V2.1 MODEL")
print("=" * 60)


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
        int(X_train.shape[1]),

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
        int(len(train_df)),

    "validation_samples":
        int(len(validation_df)),

    "training_datasets": [
        "HC3",
        "RAID",
        "SentenceAI"
    ],

    "decision_threshold":
        float(best_threshold),

    "length_targets":
        TARGETS,

    "length_weights":
        length_weights,

    "dataset_weights":
        dataset_weights,

    "model_version":
        "text_final_v21"
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

    "balanced_threshold_objective":
        float(best_balanced_score),

    "training_samples":
        int(len(train_df)),

    "validation_samples":
        int(len(validation_df))
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

threshold_df = pd.DataFrame(
    threshold_results
)


threshold_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "threshold_results.csv"
    ),
    index=False
)


# =========================================================
# SAVE LENGTH METRICS
# =========================================================

length_metrics_df = pd.DataFrame(
    length_metrics
)


length_metrics_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "length_metrics.csv"
    ),
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("V2.1 MODEL SAVED")
print("=" * 60)


print(
    "Directory:",
    MODEL_DIR
)


print()
print("Files:")


for filename in [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "linguistic_scaler.pkl",
    "text_classifier.pkl",
    "text_config.pkl",
    "text_metrics.pkl",
    "training_distribution.csv",
    "validation_results.csv",
    "threshold_results.csv",
    "length_metrics.csv"
]:

    print(
        " -",
        filename
    )


print()
print("=" * 60)
print("GENERALIZED TEXT DETECTOR V2.1 TRAINING COMPLETED")
print("=" * 60)
print()