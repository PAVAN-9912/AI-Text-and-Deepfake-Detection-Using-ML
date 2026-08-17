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


print()
print("FINAL AI TEXT DETECTOR")
print("======================")
print("HC3 + RAID + SHORT-TEXT FUSION")
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
    r"models\text_final"
)


# =========================================================
# TF-IDF SETTINGS
# =========================================================

WORD_MAX_FEATURES = 150000

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


def add_length_group(df):

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

        else:
            return "long"


    df = df.copy()

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


        # -------------------------------------------------
        # Sentence estimation
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
        #
        # This prevents text length from becoming an
        # excessively dominant feature.
        # -------------------------------------------------

        log_word_count = np.log1p(
            word_count
        )


        # -------------------------------------------------
        # Length bucket
        #
        # 0 = very short
        # 1 = short
        # 2 = medium
        # 3 = long
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


print(
    f"HC3 training samples: "
    f"{len(hc3_train)}"
)

print(
    f"HC3 validation samples: "
    f"{len(hc3_validation)}"
)


# =========================================================
# HC3 SOURCE BALANCING
# =========================================================

print()
print(
    "BUILDING SOURCE-BALANCED HC3"
)
print(
    "============================"
)


hc3_balanced_parts = []


for source in sorted(
    hc3_train["source"].unique()
):

    source_df = hc3_train[
        hc3_train["source"] == source
    ]


    human = source_df[
        source_df["label"] == 0
    ]

    ai = source_df[
        source_df["label"] == 1
    ]


    target = min(
        len(human),
        len(ai)
    )


    if target == 0:
        continue


    human_sample = human.sample(
        n=target,
        random_state=RANDOM_SEED
    )

    ai_sample = ai.sample(
        n=target,
        random_state=RANDOM_SEED
    )


    hc3_balanced_parts.append(
        human_sample
    )

    hc3_balanced_parts.append(
        ai_sample
    )


    print(
        f"{source:12s} "
        f"Human={target:5d} "
        f"AI={target:5d}"
    )


hc3_balanced = pd.concat(
    hc3_balanced_parts,
    ignore_index=True
)


print()
print(
    f"Balanced HC3 samples: "
    f"{len(hc3_balanced)}"
)


# =========================================================
# LOAD RAID
# =========================================================

print()
print("LOADING RAID DOMAIN DATA")
print("========================")


raid = pd.read_csv(
    RAID_TRAIN
)

raid = clean_dataframe(
    raid
)


print(
    f"RAID samples available: "
    f"{len(raid)}"
)


# =========================================================
# BALANCE RAID
# =========================================================

print()
print("BALANCING RAID DATA")
print("===================")


raid_human = raid[
    raid["label"] == 0
].copy()


raid_ai = raid[
    raid["label"] == 1
].copy()


# We want exactly the same number of
# Human and AI RAID samples.

RAID_TARGET_PER_LABEL = min(
    len(raid_human),
    len(raid_ai)
)


# To retain generator diversity,
# sample AI equally across generators.

AI_GENERATORS = [
    "gpt2",
    "llama-chat",
    "mpt",
    "mpt-chat"
]


raid_ai_parts = []


per_generator = (
    RAID_TARGET_PER_LABEL //
    len(AI_GENERATORS)
)


for generator in AI_GENERATORS:

    generator_df = raid_ai[
        raid_ai["raid_model"] ==
        generator
    ]


    take = min(
        per_generator,
        len(generator_df)
    )


    if take > 0:

        sampled = generator_df.sample(
            n=take,
            random_state=RANDOM_SEED
        )

        raid_ai_parts.append(
            sampled
        )


raid_ai_balanced = pd.concat(
    raid_ai_parts,
    ignore_index=True
)


# If rounding caused a small difference,
# fill from remaining AI samples.

remaining_ai_needed = (
    RAID_TARGET_PER_LABEL
    -
    len(raid_ai_balanced)
)


if remaining_ai_needed > 0:

    already_used = set(
        raid_ai_balanced.index
    )

    remaining_pool = raid_ai[
        ~raid_ai.index.isin(
            already_used
        )
    ]


    if len(remaining_pool) >= remaining_ai_needed:

        extra = remaining_pool.sample(
            n=remaining_ai_needed,
            random_state=RANDOM_SEED
        )

        raid_ai_balanced = pd.concat(
            [
                raid_ai_balanced,
                extra
            ],
            ignore_index=True
        )


raid_human_balanced = raid_human.sample(
    n=RAID_TARGET_PER_LABEL,
    random_state=RANDOM_SEED
)


raid_balanced = pd.concat(
    [
        raid_human_balanced,
        raid_ai_balanced
    ],
    ignore_index=True
)


print()
print(
    f"Balanced RAID samples: "
    f"{len(raid_balanced)}"
)


print()
print("RAID MODEL DISTRIBUTION")
print("=======================")

print(
    raid_balanced[
        "raid_model"
    ]
    .value_counts()
    .to_string()
)


print()
print("RAID LABEL DISTRIBUTION")
print("=======================")

print(
    raid_balanced[
        "label_name"
    ]
    .value_counts()
    .to_string()
)


# =========================================================
# LOAD SENTENCE DATA
# =========================================================

print()
print("LOADING SHORT-TEXT DATA")
print("=======================")


sentence_train = pd.read_csv(
    SENTENCE_TRAIN
)

sentence_train = clean_dataframe(
    sentence_train
)


print(
    f"SentenceAI samples: "
    f"{len(sentence_train)}"
)


print()
print("SENTENCEAI LABEL DISTRIBUTION")
print("=============================")

print(
    sentence_train[
        "label_name"
    ]
    .value_counts()
    .to_string()
)


# =========================================================
# ADD LENGTH GROUPS
# =========================================================

hc3_balanced = add_length_group(
    hc3_balanced
)

raid_balanced = add_length_group(
    raid_balanced
)

sentence_train = add_length_group(
    sentence_train
)

hc3_validation = add_length_group(
    hc3_validation
)


# =========================================================
# STANDARDIZE COLUMNS
# =========================================================

hc3_balanced["dataset"] = "HC3"

raid_balanced["dataset"] = "RAID"

sentence_train["dataset"] = "SentenceAI"


# =========================================================
# COMBINE TRAINING DATA
# =========================================================

print()
print("COMBINING TRAINING DATA")
print("=======================")


training_columns = [
    "text",
    "label",
    "dataset",
    "length_group"
]


hc3_part = hc3_balanced[
    training_columns
].copy()


raid_part = raid_balanced[
    training_columns
].copy()


sentence_part = sentence_train[
    training_columns
].copy()


train_df = pd.concat(
    [
        hc3_part,
        raid_part,
        sentence_part
    ],
    ignore_index=True
)


# =========================================================
# SHUFFLE
# =========================================================

train_df = train_df.sample(
    frac=1.0,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


# =========================================================
# VALIDATION
# =========================================================

validation_df = hc3_validation[
    [
        "text",
        "label",
        "length_group"
    ]
].copy()


print()
print(
    f"Final training samples: "
    f"{len(train_df)}"
)

print(
    f"HC3 validation samples: "
    f"{len(validation_df)}"
)


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print()
print("FINAL TRAIN LABEL DISTRIBUTION")
print("==============================")

print(
    train_df[
        "label"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# DATASET DISTRIBUTION
# =========================================================

print()
print("TRAINING DATASET DISTRIBUTION")
print("=============================")

print(
    pd.crosstab(
        train_df["dataset"],
        train_df["label"]
    )
    .to_string()
)


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("TRAINING LENGTH DISTRIBUTION")
print("============================")

print(
    pd.crosstab(
        train_df["length_group"],
        train_df["label"]
    )
    .to_string()
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


X_train_word = word_vectorizer.fit_transform(
    train_df["text"]
)


print(
    "Transforming validation..."
)


X_validation_word = word_vectorizer.transform(
    validation_df["text"]
)


print(
    f"Training word shape: "
    f"{X_train_word.shape}"
)

print(
    f"Validation word shape: "
    f"{X_validation_word.shape}"
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


X_train_char = char_vectorizer.fit_transform(
    train_df["text"]
)


print(
    "Transforming validation..."
)


X_validation_char = char_vectorizer.transform(
    validation_df["text"]
)


print(
    f"Training character shape: "
    f"{X_train_char.shape}"
)

print(
    f"Validation character shape: "
    f"{X_validation_char.shape}"
)


# =========================================================
# MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")


print(
    f"Transformer: "
    f"{TRANSFORMER_NAME}"
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
    f"Training embedding shape: "
    f"{X_train_embedding.shape}"
)


# =========================================================
# VALIDATION EMBEDDINGS
# =========================================================

print()
print(
    "CREATING VALIDATION EMBEDDINGS"
)
print(
    "=============================="
)


X_validation_embedding = embedder.encode(
    validation_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    f"Validation embedding shape: "
    f"{X_validation_embedding.shape}"
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
    f"Training linguistic shape: "
    f"{X_train_linguistic_raw.shape}"
)

print(
    f"Validation linguistic shape: "
    f"{X_validation_linguistic_raw.shape}"
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
# CONVERT EMBEDDINGS TO SPARSE
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
    f"Combined training shape: "
    f"{X_train.shape}"
)

print(
    f"Combined validation shape: "
    f"{X_validation.shape}"
)


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print()
print("TRAINING FINAL FUSION CLASSIFIER")
print("================================")


y_train = train_df[
    "label"
].astype(int).values


y_validation = validation_df[
    "label"
].astype(int).values


classifier = LogisticRegression(
    max_iter=2000,
    C=2.0,
    solver="liblinear",
    class_weight="balanced",
    random_state=RANDOM_SEED
)


classifier.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

print()
print("EVALUATING HC3 VALIDATION")
print("=========================")


validation_predictions = (
    classifier.predict(
        X_validation
    )
)


validation_probabilities = (
    classifier.predict_proba(
        X_validation
    )[:, 1]
)


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


print()
print("FINAL MODEL RESULTS")
print("===================")

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
# VALIDATION BY LENGTH
# =========================================================

print()
print("VALIDATION PERFORMANCE BY LENGTH")
print("================================")


validation_result_df = validation_df.copy()

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
        ] == group
    ]


    if len(subset) == 0:
        continue


    group_accuracy = accuracy_score(
        subset["label"],
        subset["prediction"]
    )


    group_f1 = f1_score(
        subset["label"],
        subset["prediction"],
        zero_division=0
    )


    print(
        f"{group:12s} "
        f"Accuracy={group_accuracy:.4f} "
        f"F1={group_f1:.4f} "
        f"Samples={len(subset)}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

print()
print("SAVING FINAL MODEL")
print("==================")


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


config = {
    "transformer": TRANSFORMER_NAME,
    "word_max_features": WORD_MAX_FEATURES,
    "char_max_features": CHAR_MAX_FEATURES,
    "feature_count": X_train.shape[1],
    "linguistic_features": [
        "log_word_count",
        "sentence_count",
        "avg_sentence_length",
        "vocabulary_diversity",
        "length_bucket"
    ],
    "random_seed": RANDOM_SEED,
    "training_samples": len(train_df),
    "validation_samples": len(validation_df),
    "training_datasets": [
        "HC3",
        "RAID_domain_balanced",
        "Human_vs_AI_Sentences"
    ]
}


save_pickle(
    config,
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "training_samples": len(train_df),
    "validation_samples": len(validation_df)
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
        "hc3_validation_results.csv"
    ),
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("FINAL MODEL SAVED")
print("=================")

print(
    f"Directory: {MODEL_DIR}"
)

print(
    "Files:"
)

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
    " - hc3_validation_results.csv"
)


print()
print(
    "FINAL TEXT DETECTOR TRAINING COMPLETED"
)