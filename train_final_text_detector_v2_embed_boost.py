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

from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer


# =========================================================
# GENERALIZED AI TEXT DETECTOR V3.1.1
# =========================================================

print()
print("=" * 70)
print("GENERALIZED AI TEXT DETECTOR V3.1.1")
print("=" * 70)
print("HC3 + RAID + SentenceAI")
print("TF-IDF + CHARACTER + MiniLM + LINGUISTIC")
print("SHORT-TEXT EMBEDDING INTERACTION")
print("LEAKAGE-SAFE THRESHOLD CALIBRATION")
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


# =========================================================
# DATASET PATHS
# =========================================================

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


# =========================================================
# MODEL DIRECTORY
# =========================================================

MODEL_DIR = (
    r"models\text_final_v3"
)


# =========================================================
# TF-IDF SETTINGS
# =========================================================

WORD_MAX_FEATURES = 150000

CHAR_MAX_FEATURES = 150000


# =========================================================
# TRAINING TARGETS
# =========================================================

TARGETS = {

    "very_short": 1500,

    "short": 2500,

    "medium": 1500,

    "long": 500

}


# =========================================================
# VALIDATION SETTINGS
# =========================================================

FINAL_VALIDATION_FRACTION = 0.10

CALIBRATION_FRACTION = 0.10


# =========================================================
# SHORT-TEXT EMBEDDING SETTINGS
# =========================================================

# Extra embedding interaction block.
#
# This gives the linear classifier an explicit representation
# of MiniLM features when the text is short.
#
# Very short text receives the strongest interaction.
# Short text receives a moderate interaction.
# Medium/long text receives smaller interaction.

SHORT_EMBEDDING_MULTIPLIERS = {

    "very_short": 1.50,

    "short": 1.25,

    "medium": 0.75,

    "long": 0.50

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


# =========================================================

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


# =========================================================

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


# =========================================================

def add_length_group(df):

    df = df.copy()

    df["length_group"] = (
        df["text"]
        .apply(get_length_group)
    )

    return df


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

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


# =========================================================
# SAVE PICKLE
# =========================================================

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
# BALANCED SAMPLING
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


    if available >= target:

        return subset.sample(
            n=target,
            random_state=RANDOM_SEED
        )


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


# SentenceAI uses label_name.

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
        ).to_string()
    )


# =========================================================
# BUILD LENGTH-AWARE TRAINING DATA
# =========================================================

print()
print("BUILDING LENGTH-AWARE TRAINING DATA")
print("===================================")


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
    # HC3 + RAID
    # =====================================================

    remaining_target = (
        target -
        sentence_target
    )


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


        # -------------------------------------------------
        # Preserve RAID generator diversity for AI.
        # -------------------------------------------------

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


                if len(sampled) < raid_target:

                    selected_texts = set(
                        sampled["text"]
                    )


                    remaining_pool = subset[
                        ~subset["text"].isin(
                            selected_texts
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

            if len(subset) > 0:

                sampled = subset.sample(

                    n=min(
                        raid_target,
                        len(subset)
                    ),

                    random_state=RANDOM_SEED

                )

            else:

                sampled = pd.DataFrame()


        if len(sampled) > 0:

            sampled = sampled.copy()

            sampled["dataset"] = "RAID"

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


# Remove duplicate text.

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
# FINAL TRAINING DISTRIBUTION
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
# FINAL HELD-OUT VALIDATION
# =========================================================
#
# This validation set is NEVER used for threshold selection.
#
# Threshold selection happens on a separate calibration set.
# =========================================================

print()
print("BUILDING FINAL HELD-OUT VALIDATION")
print("==================================")


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


    for label_value in sorted(
        dataset_df["label"].unique()
    ):

        for length_value in [

            "very_short",
            "short",
            "medium",
            "long"

        ]:

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
                    len(group_df)
                    *
                    FINAL_VALIDATION_FRACTION
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


            validation_parts.append(
                sampled
            )


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)


validation_df = validation_df.reset_index(
    drop=True
)


# =========================================================
# REMOVE VALIDATION FROM TRAINING
# =========================================================

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
    "Training samples after validation split:",
    len(train_df)
)


print(
    "Final held-out validation samples:",
    len(validation_df)
)


print()
print("VALIDATION DISTRIBUTION")
print("=======================")


print(
    pd.crosstab(
        validation_df["length_group"],
        validation_df["label"]
    ).to_string()
)


# =========================================================
# CALIBRATION SPLIT
# =========================================================
#
# A second subset is taken from training.
#
# It is used ONLY to choose the threshold.
#
# The final validation set remains untouched.
# =========================================================

print()
print("BUILDING THRESHOLD CALIBRATION SET")
print("==================================")


train_indices = np.arange(
    len(train_df)
)


stratification_key = (
    train_df["label"].astype(str)
    + "_"
    + train_df["length_group"].astype(str)
)


try:

    core_idx, calibration_idx = train_test_split(

        train_indices,

        test_size=CALIBRATION_FRACTION,

        random_state=RANDOM_SEED,

        stratify=stratification_key

    )

except ValueError:

    core_idx, calibration_idx = train_test_split(

        train_indices,

        test_size=CALIBRATION_FRACTION,

        random_state=RANDOM_SEED

    )


model_train_df = (
    train_df.iloc[core_idx]
    .reset_index(drop=True)
)


calibration_df = (
    train_df.iloc[calibration_idx]
    .reset_index(drop=True)
)


print(
    "Classifier training samples:",
    len(model_train_df)
)


print(
    "Calibration samples:",
    len(calibration_df)
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
        model_train_df["text"]
    )
)


X_calibration_word = (
    word_vectorizer.transform(
        calibration_df["text"]
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
    "Calibration word shape:",
    X_calibration_word.shape
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
        model_train_df["text"]
    )
)


X_calibration_char = (
    char_vectorizer.transform(
        calibration_df["text"]
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
    "Calibration character shape:",
    X_calibration_char.shape
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
# EMBEDDING FUNCTION
# =========================================================

def encode_texts(texts, name):

    print()
    print(name)
    print("-" * len(name))


    embeddings = embedder.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True

    )


    return embeddings.astype(
        np.float32
    )


# =========================================================
# TRAIN EMBEDDINGS
# =========================================================

X_train_embedding = encode_texts(

    model_train_df["text"].tolist(),

    "CREATING TRAINING EMBEDDINGS"

)


print(
    "Training embedding shape:",
    X_train_embedding.shape
)


# =========================================================
# CALIBRATION EMBEDDINGS
# =========================================================

X_calibration_embedding = encode_texts(

    calibration_df["text"].tolist(),

    "CREATING CALIBRATION EMBEDDINGS"

)


print(
    "Calibration embedding shape:",
    X_calibration_embedding.shape
)


# =========================================================
# VALIDATION EMBEDDINGS
# =========================================================

X_validation_embedding = encode_texts(

    validation_df["text"].tolist(),

    "CREATING VALIDATION EMBEDDINGS"

)


print(
    "Validation embedding shape:",
    X_validation_embedding.shape
)


# =========================================================
# SHORT-TEXT EMBEDDING INTERACTION
# =========================================================
#
# This is the main V3 change.
#
# Instead of asking logistic regression to discover that
# MiniLM is particularly useful when TF-IDF becomes sparse,
# we explicitly create another embedding representation
# whose magnitude depends on text length.
#
# This does NOT change the original embedding.
# It adds an additional learnable feature block.
# =========================================================

def build_short_embedding_block(
    embeddings,
    dataframe
):

    multipliers = (
        dataframe["length_group"]
        .map(
            SHORT_EMBEDDING_MULTIPLIERS
        )
        .astype(np.float32)
        .values
    )


    multipliers = (
        multipliers
        .reshape(-1, 1)
    )


    return (
        embeddings
        *
        multipliers
    ).astype(
        np.float32
    )


print()
print("BUILDING SHORT-TEXT EMBEDDING INTERACTION")
print("==========================================")


X_train_short_embedding = (
    build_short_embedding_block(
        X_train_embedding,
        model_train_df
    )
)


X_calibration_short_embedding = (
    build_short_embedding_block(
        X_calibration_embedding,
        calibration_df
    )
)


X_validation_short_embedding = (
    build_short_embedding_block(
        X_validation_embedding,
        validation_df
    )
)


print(
    "Short interaction shape:",
    X_train_short_embedding.shape
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_train_linguistic_raw = (
    build_linguistic_features(
        model_train_df["text"].tolist()
    )
)


X_calibration_linguistic_raw = (
    build_linguistic_features(
        calibration_df["text"].tolist()
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
    "Calibration linguistic shape:",
    X_calibration_linguistic_raw.shape
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


X_calibration_linguistic = (
    linguistic_scaler.transform(
        X_calibration_linguistic_raw
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


X_calibration_embedding_sparse = csr_matrix(
    X_calibration_embedding
)


X_validation_embedding_sparse = csr_matrix(
    X_validation_embedding
)


X_train_short_embedding_sparse = csr_matrix(
    X_train_short_embedding
)


X_calibration_short_embedding_sparse = csr_matrix(
    X_calibration_short_embedding
)


X_validation_short_embedding_sparse = csr_matrix(
    X_validation_short_embedding
)


X_train_linguistic_sparse = csr_matrix(
    X_train_linguistic
)


X_calibration_linguistic_sparse = csr_matrix(
    X_calibration_linguistic
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

        X_train_short_embedding_sparse,

        X_train_linguistic_sparse

    ],

    format="csr"

)


X_calibration = hstack(

    [

        X_calibration_word,

        X_calibration_char,

        X_calibration_embedding_sparse,

        X_calibration_short_embedding_sparse,

        X_calibration_linguistic_sparse

    ],

    format="csr"

)


X_validation = hstack(

    [

        X_validation_word,

        X_validation_char,

        X_validation_embedding_sparse,

        X_validation_short_embedding_sparse,

        X_validation_linguistic_sparse

    ],

    format="csr"

)


print(
    "Combined training shape:",
    X_train.shape
)


print(
    "Combined calibration shape:",
    X_calibration.shape
)


print(
    "Combined validation shape:",
    X_validation.shape
)


# =========================================================
# LABELS
# =========================================================

y_train = (
    model_train_df["label"]
    .astype(int)
    .values
)


y_calibration = (
    calibration_df["label"]
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

    model_train_df[
        "length_group"
    ]

    .map(length_weights)

    .astype(float)

    .values

)


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print()
print("TRAINING V3 FUSION CLASSIFIER")
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


print()
print("Training completed.")


# =========================================================
# CALIBRATION PROBABILITIES
# =========================================================

print()
print("CALIBRATION PROBABILITIES")
print("=========================")


calibration_probabilities = (

    classifier

    .predict_proba(
        X_calibration
    )[:, 1]

)


# =========================================================
# LENGTH-AWARE THRESHOLD CALIBRATION (V3.1)
# =========================================================
#
# IMPORTANT:
#
# The final validation set is NOT used here.
#
# Thresholds are selected ONLY on the separate calibration set
# by optimizing F1 separately for each length group.
# =========================================================

print()
print("CALIBRATING LENGTH-AWARE DECISION THRESHOLDS")
print("===========================================")

LENGTH_GROUPS = [
    "very_short",
    "short",
    "medium",
    "long"
]

length_thresholds = {}
threshold_results = []

for group in LENGTH_GROUPS:
    group_mask = (calibration_df["length_group"] == group).values
    y_cal_g = y_calibration[group_mask]
    probs_cal_g = calibration_probabilities[group_mask]

    group_best_thresh = 0.50
    group_best_f1 = -1.0
    group_best_acc = 0.0
    group_best_prec = 0.0
    group_best_rec = 0.0

    for threshold in np.arange(0.20, 0.81, 0.01):
        thresh_val = round(float(threshold), 2)
        preds = (probs_cal_g >= thresh_val).astype(int)

        current_f1 = f1_score(
            y_cal_g,
            preds,
            zero_division=0
        )
        current_acc = accuracy_score(
            y_cal_g,
            preds
        )
        current_prec = precision_score(
            y_cal_g,
            preds,
            zero_division=0
        )
        current_rec = recall_score(
            y_cal_g,
            preds,
            zero_division=0
        )

        threshold_results.append(
            {
                "length_group": group,
                "threshold": float(thresh_val),
                "samples": int(len(y_cal_g)),
                "accuracy": float(current_acc),
                "precision": float(current_prec),
                "recall": float(current_rec),
                "f1": float(current_f1)
            }
        )

        if current_f1 > group_best_f1:
            group_best_f1 = current_f1
            group_best_acc = current_acc
            group_best_prec = current_prec
            group_best_rec = current_rec
            group_best_thresh = thresh_val
        elif abs(current_f1 - group_best_f1) < 1e-7:
            if current_acc > group_best_acc:
                group_best_acc = current_acc
                group_best_prec = current_prec
                group_best_rec = current_rec
                group_best_thresh = thresh_val
            elif abs(current_acc - group_best_acc) < 1e-7:
                if abs(thresh_val - 0.50) < abs(group_best_thresh - 0.50):
                    group_best_prec = current_prec
                    group_best_rec = current_rec
                    group_best_thresh = thresh_val

    length_thresholds[group] = float(group_best_thresh)
    print(
        f"Group: {group:12s} | N={len(y_cal_g):3d} | "
        f"Selected Threshold: {group_best_thresh:.2f} | "
        f"Calibration F1: {group_best_f1:.4f} | "
        f"Accuracy: {group_best_acc:.4f} | "
        f"Precision: {group_best_prec:.4f} | "
        f"Recall: {group_best_rec:.4f}"
    )

print()
print("FROZEN LENGTH-AWARE THRESHOLDS:")
for grp, thresh in length_thresholds.items():
    print(f" - {grp:12s}: {thresh:.2f}")


# =========================================================
# FINAL HELD-OUT VALIDATION
# =========================================================
#
# The frozen calibration thresholds are applied exactly once
# to the untouched final validation set.
# =========================================================

print()
print("FINAL HELD-OUT VALIDATION")
print("=========================")

validation_probabilities = (
    classifier
    .predict_proba(
        X_validation
    )[:, 1]
)

applied_thresholds = np.array([
    length_thresholds.get(grp, 0.50)
    for grp in validation_df["length_group"]
])

validation_predictions = (
    validation_probabilities >= applied_thresholds
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


cm = confusion_matrix(

    y_validation,

    validation_predictions

)


# =========================================================
# ERROR COUNTS
# =========================================================

tn, fp, fn, tp = cm.ravel()


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 70)
print("V3.1 FINAL VALIDATION RESULTS")
print("=" * 70)


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
    f"Thresholds: {length_thresholds}"
)


print()
print("ERROR COUNTS")
print("============")


print(
    f"True Negative : {tn}"
)


print(
    f"False Positive: {fp}"
)


print(
    f"False Negative: {fn}"
)


print(
    f"True Positive : {tp}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print("================")


print(cm)


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
# VALIDATION RESULT DATAFRAME
# =========================================================

validation_result_df = (
    validation_df.copy()
)


validation_result_df[
    "probability"
] = validation_probabilities


validation_result_df[
    "applied_threshold"
] = applied_thresholds


validation_result_df[
    "prediction"
] = validation_predictions


validation_result_df[
    "error_type"
] = "Correct"


validation_result_df.loc[

    (
        validation_result_df["label"] == 0
    )

    &

    (
        validation_result_df["prediction"] == 1
    ),

    "error_type"

] = "False Positive"


validation_result_df.loc[

    (
        validation_result_df["label"] == 1
    )

    &

    (
        validation_result_df["prediction"] == 0
    ),

    "error_type"

] = "False Negative"


# =========================================================
# PERFORMANCE BY LENGTH
# =========================================================

print()
print("VALIDATION PERFORMANCE BY LENGTH")
print("================================")


length_results = []


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


    group_fp = (

        (subset["label"] == 0)

        &

        (subset["prediction"] == 1)

    ).sum()


    group_fn = (

        (subset["label"] == 1)

        &

        (subset["prediction"] == 0)

    ).sum()


    length_results.append(

        {

            "length_group":
                group,

            "samples":
                len(subset),

            "accuracy":
                group_accuracy,

            "precision":
                group_precision,

            "recall":
                group_recall,

            "f1":
                group_f1,

            "false_positive":
                int(group_fp),

            "false_negative":
                int(group_fn)

        }

    )


    print(

        f"{group:12s} "

        f"N={len(subset):4d} "

        f"Accuracy={group_accuracy:.4f} "

        f"Precision={group_precision:.4f} "

        f"Recall={group_recall:.4f} "

        f"F1={group_f1:.4f} "

        f"FP={group_fp:3d} "

        f"FN={group_fn:3d}"

    )


length_results_df = pd.DataFrame(
    length_results
)


# =========================================================
# ERROR SUMMARY
# =========================================================

print()
print("ERROR SUMMARY")
print("=============")


print(
    validation_result_df[
        "error_type"
    ]
    .value_counts()
    .to_string()
)


# =========================================================
# SAVE MODEL
# =========================================================

print()
print("SAVING V3.1 MODEL")
print("===============")


os.makedirs(

    MODEL_DIR,

    exist_ok=True

)


# =========================================================
# SAVE VECTORIZERS
# =========================================================

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


# =========================================================
# SAVE SCALER
# =========================================================

save_pickle(

    linguistic_scaler,

    os.path.join(

        MODEL_DIR,

        "linguistic_scaler.pkl"

    )

)


# =========================================================
# SAVE CLASSIFIER
# =========================================================

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


    "base_embedding_dimension":
        int(
            X_train_embedding.shape[1]
        ),


    "short_embedding_dimension":
        int(
            X_train_short_embedding.shape[1]
        ),


    "linguistic_features": [

        "log_word_count",

        "sentence_count",

        "avg_sentence_length",

        "vocabulary_diversity",

        "length_bucket"

    ],


    "short_embedding_multipliers":
        SHORT_EMBEDDING_MULTIPLIERS,


    "length_weights":
        length_weights,


    "random_seed":
        RANDOM_SEED,


    "training_samples":
        len(model_train_df),


    "calibration_samples":
        len(calibration_df),


    "validation_samples":
        len(validation_df),


    "training_datasets": [

        "HC3",

        "RAID",

        "SentenceAI"

    ],


    "decision_threshold":
        length_thresholds,


    "length_thresholds":
        length_thresholds,


    "length_targets":
        TARGETS,


    "model_version":
        "text_final_v3.1"

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
        length_thresholds,


    "length_thresholds":
        length_thresholds,

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp),

    "training_samples":
        len(model_train_df),

    "calibration_samples":
        len(calibration_df),

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
# SAVE LENGTH RESULTS
# =========================================================

length_results_df.to_csv(

    os.path.join(

        MODEL_DIR,

        "length_results.csv"

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
# SAVE MODEL SUMMARY
# =========================================================

summary = {

    "model":
        "text_final_v3.1",

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

    "thresholds":
        str(length_thresholds),

    "false_positive":
        fp,

    "false_negative":
        fn,

    "training_samples":
        len(model_train_df),

    "calibration_samples":
        len(calibration_df),

    "validation_samples":
        len(validation_df)

}


pd.DataFrame(
    [summary]
).to_csv(

    os.path.join(

        MODEL_DIR,

        "model_summary.csv"

    ),

    index=False

)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 70)
print("V3.1 MODEL SAVED")
print("=" * 70)


print(
    "Directory:",
    MODEL_DIR
)


print()
print("Files:")


files_to_save = [

    "word_tfidf_vectorizer.pkl",

    "char_tfidf_vectorizer.pkl",

    "linguistic_scaler.pkl",

    "text_classifier.pkl",

    "text_config.pkl",

    "text_metrics.pkl",

    "training_distribution.csv",

    "validation_results.csv",

    "length_results.csv",

    "threshold_results.csv",

    "model_summary.csv"

]


for filename in files_to_save:

    print(
        f" - {filename}"
    )


print()
print("=" * 70)
print("GENERALIZED TEXT DETECTOR V3.1 TRAINING COMPLETED")
print("=" * 70)