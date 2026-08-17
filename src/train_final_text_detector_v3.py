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
# GENERALIZED AI TEXT DETECTOR V3
# =========================================================

print()
print("=" * 60)
print("GENERALIZED AI TEXT DETECTOR V3")
print("=" * 60)
print("HC3 + RAID + SentenceAI")
print("DATASET + LENGTH BALANCED TRAINING")
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
    r"models\text_final_v3"
)

WORD_MAX_FEATURES = 60000
CHAR_MAX_FEATURES = 150000


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


def add_length_group(df):

    df = df.copy()

    def classify_length(text):

        count = len(
            text.split()
        )

        if count <= 10:
            return "very_short"

        elif count <= 30:
            return "short"

        elif count <= 60:
            return "medium"

        return "long"

    df["length_group"] = (
        df["text"]
        .apply(classify_length)
    )

    return df


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


def print_distribution(
    df,
    name
):

    print()
    print(name)
    print("=" * len(name))

    print()

    print(
        "LABEL DISTRIBUTION"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "LENGTH DISTRIBUTION"
    )

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
        .to_string()
    )


def save_pickle(
    obj,
    path
):

    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(
            obj,
            f
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
    f"HC3 training samples: "
    f"{len(hc3_train)}"
)

print(
    f"HC3 validation samples: "
    f"{len(hc3_validation)}"
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
    f"RAID samples: "
    f"{len(raid)}"
)


# =========================================================
# LOAD SENTENCEAI
# =========================================================

print()
print("LOADING SENTENCEAI")
print("==================")

sentence = pd.read_csv(
    SENTENCE_TRAIN
)

sentence = clean_dataframe(
    sentence
)

sentence = add_length_group(
    sentence
)

print(
    f"SentenceAI samples: "
    f"{len(sentence)}"
)


# =========================================================
# STANDARDIZE DATASETS
# =========================================================

hc3_train["dataset"] = "HC3"
raid["dataset"] = "RAID"
sentence["dataset"] = "SentenceAI"


required_columns = [
    "text",
    "label",
    "dataset",
    "length_group"
]


hc3_train = hc3_train[
    required_columns
].copy()

raid = raid[
    required_columns
].copy()

sentence = sentence[
    required_columns
].copy()


# =========================================================
# ORIGINAL DISTRIBUTIONS
# =========================================================

print()
print("=" * 60)
print("ORIGINAL DATA DISTRIBUTIONS")
print("=" * 60)

print_distribution(
    hc3_train,
    "HC3"
)

print_distribution(
    raid,
    "RAID"
)

print_distribution(
    sentence,
    "SentenceAI"
)


# =========================================================
# BUILD SOURCE DATA DICTIONARY
# =========================================================

sources = {
    "HC3": hc3_train,
    "RAID": raid,
    "SentenceAI": sentence
}


# =========================================================
# LENGTH-AWARE BALANCING
# =========================================================

print()
print("=" * 60)
print("BUILDING V3 BALANCED TRAINING DATA")
print("=" * 60)


length_groups = [
    "very_short",
    "short",
    "medium",
    "long"
]


balanced_parts = []


# ---------------------------------------------------------
# TARGETS
# ---------------------------------------------------------

# Each dataset contributes a controlled amount to each
# length/label combination.
#
# We deliberately avoid letting HC3 dominate long text
# and avoid letting SentenceAI dominate short text.

TARGETS = {

    "very_short": {
        "HC3": 150,
        "RAID": 150,
        "SentenceAI": 300
    },

    "short": {
        "HC3": 400,
        "RAID": 400,
        "SentenceAI": 500
    },

    "medium": {
        "HC3": 400,
        "RAID": 300,
        "SentenceAI": 300
    },

    "long": {
        "HC3": 500,
        "RAID": 300,
        "SentenceAI": 0
    }
}


for length_group in length_groups:

    print()
    print(
        f"PROCESSING {length_group.upper()}"
    )

    print("-" * 45)


    for dataset_name in [
        "HC3",
        "RAID",
        "SentenceAI"
    ]:

        target_total = (
            TARGETS[
                length_group
            ][
                dataset_name
            ]
        )

        if target_total <= 0:
            continue


        source_df = sources[
            dataset_name
        ]


        for label in [0, 1]:

            available = source_df[
                (
                    source_df[
                        "length_group"
                    ]
                    ==
                    length_group
                )
                &
                (
                    source_df[
                        "label"
                    ]
                    ==
                    label
                )
            ].copy()


            # Half target for each label.
            target = (
                target_total
                // 2
            )


            # If source has fewer examples,
            # use everything available.
            take = min(
                target,
                len(available)
            )


            if take > 0:

                sampled = available.sample(
                    n=take,
                    random_state=(
                        RANDOM_SEED
                        +
                        len(balanced_parts)
                    )
                )

                balanced_parts.append(
                    sampled
                )


            label_name = (
                "Human"
                if label == 0
                else "AI"
            )

            print(
                f"{dataset_name:10s} "
                f"{length_group:11s} "
                f"{label_name:6s} "
                f"Available={len(available):5d} "
                f"Target={target:4d} "
                f"Used={take:4d}"
            )


# =========================================================
# COMBINE
# =========================================================

balanced_training = pd.concat(
    balanced_parts,
    ignore_index=True
)


# =========================================================
# SHUFFLE
# =========================================================

balanced_training = (
    balanced_training
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 60)
print("V3 FINAL TRAINING DISTRIBUTION")
print("=" * 60)

print(
    f"Total samples: "
    f"{len(balanced_training)}"
)

print()

print(
    "LABEL DISTRIBUTION"
)

print(
    balanced_training[
        "label"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print(
    "DATASET DISTRIBUTION"
)

print(
    pd.crosstab(
        balanced_training[
            "dataset"
        ],
        balanced_training[
            "label"
        ]
    )
    .to_string()
)

print()

print(
    "LENGTH DISTRIBUTION"
)

print(
    pd.crosstab(
        balanced_training[
            "length_group"
        ],
        balanced_training[
            "label"
        ]
    )
    .reindex(
        length_groups
    )
    .fillna(0)
    .astype(int)
    .to_string()
)


# =========================================================
# BUILD VALIDATION SET
# =========================================================

print()
print("=" * 60)
print("BUILDING INTERNAL VALIDATION SET")
print("=" * 60)


# We use a stratified-style sample from the V3 training pool.
# The validation data is never used to fit the classifier.

validation_parts = []


for length_group in length_groups:

    for label in [0, 1]:

        subset = balanced_training[
            (
                balanced_training[
                    "length_group"
                ]
                ==
                length_group
            )
            &
            (
                balanced_training[
                    "label"
                ]
                ==
                label
            )
        ]


        if len(subset) == 0:
            continue


        validation_count = max(
            20,
            int(
                len(subset) * 0.10
            )
        )


        validation_count = min(
            validation_count,
            len(subset)
        )


        sampled = subset.sample(
            n=validation_count,
            random_state=(
                RANDOM_SEED
                +
                label
                +
                len(validation_parts)
            )
        )


        validation_parts.append(
            sampled
        )


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)


# Remove validation samples from training.
training_df = (
    balanced_training
    .drop(
        validation_df.index,
        errors="ignore"
    )
    .copy()
)


# The indexes above may not correspond after concat,
# so rebuild training by removing exact text+metadata rows.

validation_keys = set(
    zip(
        validation_df["text"],
        validation_df["dataset"],
        validation_df["label"]
    )
)


training_df = balanced_training[
    ~balanced_training.apply(
        lambda row:
            (
                row["text"],
                row["dataset"],
                row["label"]
            )
            in validation_keys,
        axis=1
    )
].copy()


training_df = (
    training_df
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED
    )
    .reset_index(drop=True)
)


validation_df = (
    validation_df
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED
    )
    .reset_index(drop=True)
)


print()
print(
    f"Training samples: "
    f"{len(training_df)}"
)

print(
    f"Validation samples: "
    f"{len(validation_df)}"
)

print()

print(
    "VALIDATION LENGTH DISTRIBUTION"
)

print(
    pd.crosstab(
        validation_df[
            "length_group"
        ],
        validation_df[
            "label"
        ]
    )
    .reindex(
        length_groups
    )
    .fillna(0)
    .astype(int)
    .to_string()
)


# =========================================================
# SAVE TRAINING DISTRIBUTION
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


distribution = (
    balanced_training
    .groupby(
        [
            "dataset",
            "length_group",
            "label"
        ]
    )
    .size()
    .reset_index(
        name="samples"
    )
)


distribution.to_csv(
    os.path.join(
        MODEL_DIR,
        "training_distribution.csv"
    ),
    index=False
)


# =========================================================
# TEXT ARRAYS
# =========================================================

train_texts = (
    training_df[
        "text"
    ]
    .tolist()
)

validation_texts = (
    validation_df[
        "text"
    ]
    .tolist()
)

y_train = (
    training_df[
        "label"
    ]
    .astype(int)
    .to_numpy()
)

y_validation = (
    validation_df[
        "label"
    ]
    .astype(int)
    .to_numpy()
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

    min_df=2,

    sublinear_tf=True,

    strip_accents="unicode",

    lowercase=True
)


print(
    "Fitting word vocabulary..."
)

train_word = (
    word_vectorizer.fit_transform(
        train_texts
    )
)


print(
    "Transforming validation..."
)

validation_word = (
    word_vectorizer.transform(
        validation_texts
    )
)


print(
    "Training word shape:",
    train_word.shape
)

print(
    "Validation word shape:",
    validation_word.shape
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

    min_df=2,

    sublinear_tf=True,

    lowercase=True
)


print(
    "Fitting character vocabulary..."
)

train_char = (
    char_vectorizer.fit_transform(
        train_texts
    )
)


print(
    "Transforming validation..."
)

validation_char = (
    char_vectorizer.transform(
        validation_texts
    )
)


print(
    "Training character shape:",
    train_char.shape
)

print(
    "Validation character shape:",
    validation_char.shape
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
# TRAIN EMBEDDINGS
# =========================================================

print()
print("=" * 60)
print("CREATING TRAINING EMBEDDINGS")
print("=" * 60)


train_embeddings = transformer.encode(

    train_texts,

    batch_size=32,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


print(
    "Training embedding shape:",
    train_embeddings.shape
)


# =========================================================
# VALIDATION EMBEDDINGS
# =========================================================

print()
print("=" * 60)
print("CREATING VALIDATION EMBEDDINGS")
print("=" * 60)


validation_embeddings = transformer.encode(

    validation_texts,

    batch_size=32,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


print(
    "Validation embedding shape:",
    validation_embeddings.shape
)


train_embeddings = csr_matrix(
    train_embeddings
)

validation_embeddings = csr_matrix(
    validation_embeddings
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

print()
print("=" * 60)
print("BUILDING LINGUISTIC FEATURES")
print("=" * 60)


train_linguistic = (
    build_linguistic_features(
        train_texts
    )
)

validation_linguistic = (
    build_linguistic_features(
        validation_texts
    )
)


print(
    "Training linguistic shape:",
    train_linguistic.shape
)

print(
    "Validation linguistic shape:",
    validation_linguistic.shape
)


# =========================================================
# SCALE LINGUISTIC FEATURES
# =========================================================

print()
print("=" * 60)
print("SCALING LINGUISTIC FEATURES")
print("=" * 60)


linguistic_scaler = StandardScaler()

train_linguistic = (
    linguistic_scaler.fit_transform(
        train_linguistic
    )
)

validation_linguistic = (
    linguistic_scaler.transform(
        validation_linguistic
    )
)


train_linguistic = csr_matrix(
    train_linguistic
)

validation_linguistic = csr_matrix(
    validation_linguistic
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("=" * 60)
print("COMBINING FEATURES")
print("=" * 60)


train_features = hstack(
    [
        train_word,
        train_char,
        train_embeddings,
        train_linguistic
    ],
    format="csr"
)


validation_features = hstack(
    [
        validation_word,
        validation_char,
        validation_embeddings,
        validation_linguistic
    ],
    format="csr"
)


print(
    "Combined training shape:",
    train_features.shape
)

print(
    "Combined validation shape:",
    validation_features.shape
)


# =========================================================
# FEATURE COUNT
# =========================================================

expected_features = (
    train_word.shape[1]
    +
    train_char.shape[1]
    +
    train_embeddings.shape[1]
    +
    train_linguistic.shape[1]
)


print()
print(
    "Expected feature count:",
    expected_features
)


# =========================================================
# CLASSIFIER
# =========================================================

print()
print("=" * 60)
print("TRAINING V3 FUSION CLASSIFIER")
print("=" * 60)


classifier = LogisticRegression(

    max_iter=2000,

    C=2.0,

    class_weight="balanced",

    solver="liblinear",

    random_state=RANDOM_SEED
)


classifier.fit(
    train_features,
    y_train
)


print(
    "Training completed."
)


# =========================================================
# VALIDATION PROBABILITIES
# =========================================================

print()
print("=" * 60)
print("RUNNING VALIDATION PREDICTIONS")
print("=" * 60)


validation_probabilities = (
    classifier.predict_proba(
        validation_features
    )[:, 1]
)


# =========================================================
# THRESHOLD SEARCH
# =========================================================

print()
print("=" * 60)
print("SEARCHING DECISION THRESHOLD")
print("=" * 60)


threshold_rows = []


for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


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


    threshold_rows.append(
        {
            "threshold": float(
                threshold
            ),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


best_threshold_row = (
    threshold_df
    .sort_values(
        [
            "f1",
            "accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)


BEST_THRESHOLD = float(
    best_threshold_row[
        "threshold"
    ]
)


print()
print(
    f"Selected threshold: "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['f1']:.4f}"
)


# =========================================================
# FINAL VALIDATION PREDICTIONS
# =========================================================

validation_predictions = (
    validation_probabilities
    >= BEST_THRESHOLD
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
print("V3 VALIDATION RESULTS")
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
    f"Threshold: {BEST_THRESHOLD:.2f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print("================")

cm = confusion_matrix(
    y_validation,
    validation_predictions
)

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
# PERFORMANCE BY LENGTH
# =========================================================

print()
print("=" * 60)
print("V3 PERFORMANCE BY TEXT LENGTH")
print("=" * 60)


for length_group in length_groups:

    mask = (
        validation_df[
            "length_group"
        ]
        ==
        length_group
    )

    if mask.sum() == 0:
        continue


    true_values = (
        y_validation[mask]
    )

    predicted_values = (
        validation_predictions[mask]
    )


    group_accuracy = (
        accuracy_score(
            true_values,
            predicted_values
        )
    )

    group_precision = (
        precision_score(
            true_values,
            predicted_values,
            zero_division=0
        )
    )

    group_recall = (
        recall_score(
            true_values,
            predicted_values,
            zero_division=0
        )
    )

    group_f1 = (
        f1_score(
            true_values,
            predicted_values,
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
# SAVE MODEL
# =========================================================

print()
print("=" * 60)
print("SAVING V3 MODEL")
print("=" * 60)


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

    "model_version": "V3",

    "transformer":
        TRANSFORMER_NAME,

    "word_max_features":
        WORD_MAX_FEATURES,

    "char_max_features":
        CHAR_MAX_FEATURES,

    "word_features":
        int(
            train_word.shape[1]
        ),

    "char_features":
        int(
            train_char.shape[1]
        ),

    "minilm_features":
        int(
            train_embeddings.shape[1]
        ),

    "linguistic_features":
        5,

    "feature_count":
        int(
            expected_features
        ),

    "random_seed":
        RANDOM_SEED,

    "training_samples":
        int(
            len(training_df)
        ),

    "validation_samples":
        int(
            len(validation_df)
        ),

    "training_datasets":
        [
            "HC3",
            "RAID",
            "SentenceAI"
        ],

    "threshold":
        BEST_THRESHOLD
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

    "threshold":
        float(BEST_THRESHOLD)
}


save_pickle(
    metrics,
    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    )
)


# =========================================================
# VALIDATION RESULTS
# =========================================================

validation_results = (
    validation_df.copy()
)

validation_results[
    "ai_probability"
] = validation_probabilities

validation_results[
    "prediction"
] = validation_predictions

validation_results[
    "prediction_name"
] = np.where(
    validation_predictions == 1,
    "AI",
    "Human"
)


validation_results.to_csv(
    os.path.join(
        MODEL_DIR,
        "validation_results.csv"
    ),
    index=False
)


# =========================================================
# THRESHOLD RESULTS
# =========================================================

threshold_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "threshold_results.csv"
    ),
    index=False
)


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 60)
print("V3 MODEL SAVED")
print("=" * 60)

print(
    "Directory:",
    MODEL_DIR
)

print()
print("Files:")

for filename in sorted(
    os.listdir(
        MODEL_DIR
    )
):

    print(
        " -",
        filename
    )


print()
print("=" * 60)
print("GENERALIZED TEXT DETECTOR V3 TRAINING COMPLETED")
print("=" * 60)
print()