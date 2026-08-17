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

RAID_DIR = r"datasets\text\RAID"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("V2 MODEL — RAID EXTERNAL TEST")
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

    if "content" in df.columns:
        return "content"

    raise ValueError(
        "No text column found in RAID dataset."
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
# FIND RAID EXTERNAL DATA
# =========================================================

print("SEARCHING FOR RAID EXTERNAL DATA")
print("================================")


if not os.path.exists(RAID_DIR):

    raise FileNotFoundError(
        f"RAID directory not found: {RAID_DIR}"
    )


csv_files = []

for filename in os.listdir(RAID_DIR):

    if filename.lower().endswith(".csv"):

        csv_files.append(
            os.path.join(
                RAID_DIR,
                filename
            )
        )


if not csv_files:

    raise FileNotFoundError(
        "No CSV files found in RAID directory."
    )


candidates = []


for file_path in csv_files:

    try:

        sample_df = pd.read_csv(
            file_path,
            nrows=10
        )

        columns = set(
            sample_df.columns
        )

        score = 0

        filename = os.path.basename(
            file_path
        ).lower()

        if (
            "external" in filename
            or
            "test" in filename
        ):
            score += 10

        if (
            "text" in columns
            or
            "generation" in columns
            or
            "content" in columns
        ):
            score += 5

        if "label" in columns:
            score += 5

        if "label_name" in columns:
            score += 5

        if "raid_model" in columns:
            score += 3

        if "model" in columns:
            score += 3

        if "domain" in columns:
            score += 2

        candidates.append(
            (
                score,
                file_path
            )
        )

    except Exception:
        pass


if not candidates:

    raise RuntimeError(
        "Could not identify a RAID CSV file."
    )


candidates.sort(
    reverse=True
)


TEST_FILE = candidates[0][1]


print(
    "Selected RAID file:",
    TEST_FILE
)


# =========================================================
# LOAD RAID DATA
# =========================================================

print()
print("LOADING RAID EXTERNAL DATA")
print("==========================")

raid_df = pd.read_csv(
    TEST_FILE
)

raid_df = clean_dataframe(
    raid_df
)

raid_df["length_group"] = (
    raid_df["text"]
    .apply(get_length_group)
)


print(
    f"External samples: "
    f"{len(raid_df)}"
)


# =========================================================
# LABEL PREPARATION
# =========================================================

if "label" in raid_df.columns:

    label_series = raid_df["label"]

elif "label_name" in raid_df.columns:

    label_series = raid_df["label_name"]

else:

    raise ValueError(
        "RAID dataset does not contain "
        "label or label_name."
    )


def convert_label(value):

    if isinstance(value, str):

        value_lower = (
            value.strip()
            .lower()
        )

        if value_lower in [
            "human",
            "0"
        ]:
            return 0

        if value_lower in [
            "ai",
            "1"
        ]:
            return 1

    try:

        return int(value)

    except Exception:

        raise ValueError(
            f"Unknown RAID label: {value}"
        )


y_true = np.asarray(
    [
        convert_label(value)
        for value in label_series
    ],
    dtype=int
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


# =========================================================
# MODEL DISTRIBUTION
# =========================================================

model_column = None

if "raid_model" in raid_df.columns:

    model_column = "raid_model"

elif "model" in raid_df.columns:

    model_column = "model"


if model_column is not None:

    print()
    print("MODEL DISTRIBUTION")
    print("==================")

    print(
        raid_df[
            model_column
        ]
        .value_counts()
        .to_string()
    )


# =========================================================
# DOMAIN DISTRIBUTION
# =========================================================

if "domain" in raid_df.columns:

    print()
    print("DOMAIN DISTRIBUTION")
    print("===================")

    print(
        raid_df[
            "domain"
        ]
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
        raid_df["length_group"],
        pd.Series(
            y_true,
            index=raid_df.index,
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


print(
    "V2 model components loaded."
)


# =========================================================
# WORD TF-IDF
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")

word_features = (
    word_vectorizer.transform(
        raid_df["text"]
    )
)

print(
    "Word feature shape:",
    word_features.shape
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")

char_features = (
    char_vectorizer.transform(
        raid_df["text"]
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
    raid_df["text"].tolist(),
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
        raid_df["text"].tolist()
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


if (
    expected_features
    !=
    classifier.n_features_in_
):

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


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("V2 RAID EXTERNAL RESULTS")
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
# PERFORMANCE BY GENERATOR
# =========================================================

if model_column is not None:

    print()
    print("PERFORMANCE BY RAID GENERATOR")
    print("=============================")

    for generator in sorted(
        raid_df[
            model_column
        ]
        .dropna()
        .astype(str)
        .unique()
    ):

        mask = (
            raid_df[
                model_column
            ]
            .astype(str)
            .to_numpy()
            ==
            generator
        )

        if mask.sum() == 0:
            continue

        group_y = y_true[mask]

        group_pred = (
            predictions[mask]
        )

        group_accuracy = (
            accuracy_score(
                group_y,
                group_pred
            )
        )

        group_f1 = (
            f1_score(
                group_y,
                group_pred,
                zero_division=0
            )
        )

        print(
            f"{generator:14s} "
            f"Accuracy={group_accuracy:.4f} "
            f"F1={group_f1:.4f} "
            f"Samples={mask.sum()}"
        )


# =========================================================
# AI RECALL BY GENERATOR
# =========================================================

if model_column is not None:

    print()
    print("AI DETECTION BY GENERATOR")
    print("=========================")

    for generator in sorted(
        raid_df[
            model_column
        ]
        .dropna()
        .astype(str)
        .unique()
    ):

        mask = (
            (
                raid_df[
                    model_column
                ]
                .astype(str)
                .to_numpy()
                ==
                generator
            )
            &
            (y_true == 1)
        )

        if mask.sum() == 0:
            continue

        ai_recall = (
            recall_score(
                y_true[mask],
                predictions[mask],
                zero_division=0
            )
        )

        print(
            f"{generator:14s} "
            f"AI Recall={ai_recall:.4f} "
            f"AI Samples={mask.sum()}"
        )


# =========================================================
# PERFORMANCE BY DOMAIN
# =========================================================

if "domain" in raid_df.columns:

    print()
    print("PERFORMANCE BY DOMAIN")
    print("=====================")

    for domain in sorted(
        raid_df[
            "domain"
        ]
        .dropna()
        .astype(str)
        .unique()
    ):

        mask = (
            raid_df[
                "domain"
            ]
            .astype(str)
            .to_numpy()
            ==
            domain
        )

        if mask.sum() == 0:
            continue

        group_y = y_true[mask]

        group_pred = (
            predictions[mask]
        )

        group_accuracy = (
            accuracy_score(
                group_y,
                group_pred
            )
        )

        group_f1 = (
            f1_score(
                group_y,
                group_pred,
                zero_division=0
            )
        )

        print(
            f"{domain:14s} "
            f"Accuracy={group_accuracy:.4f} "
            f"F1={group_f1:.4f} "
            f"Samples={mask.sum()}"
        )


# =========================================================
# PERFORMANCE BY TEXT LENGTH
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
        raid_df[
            "length_group"
        ]
        .to_numpy()
        ==
        length_group
    )

    if mask.sum() == 0:
        continue

    group_y = y_true[mask]

    group_pred = (
        predictions[mask]
    )

    group_accuracy = (
        accuracy_score(
            group_y,
            group_pred
        )
    )

    group_precision = (
        precision_score(
            group_y,
            group_pred,
            zero_division=0
        )
    )

    group_recall = (
        recall_score(
            group_y,
            group_pred,
            zero_division=0
        )
    )

    group_f1 = (
        f1_score(
            group_y,
            group_pred,
            zero_division=0
        )
    )

    print(
        f"{length_group:14s} "
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
            raid_df[
                "length_group"
            ]
            .to_numpy()
            ==
            length_group
        )
        &
        (y_true == 1)
    )

    if mask.sum() == 0:
        continue

    ai_recall = (
        recall_score(
            y_true[mask],
            predictions[mask],
            zero_division=0
        )
    )

    print(
        f"{length_group:14s} "
        f"AI Recall={ai_recall:.4f} "
        f"AI Samples={mask.sum()}"
    )


# =========================================================
# SAVE DETAILED RESULTS
# =========================================================

results_df = raid_df.copy()

results_df["true_label"] = (
    y_true
)

results_df["predicted_label"] = (
    predictions
)

results_df["true_label_name"] = (
    results_df[
        "true_label"
    ]
    .map({
        0: "Human",
        1: "AI"
    })
)

results_df["predicted_label_name"] = (
    results_df[
        "predicted_label"
    ]
    .map({
        0: "Human",
        1: "AI"
    })
)

results_df["ai_probability"] = (
    probabilities
)

results_df["human_probability"] = (
    1.0 - probabilities
)


OUTPUT_FILE = (
    r"datasets\text\RAID"
    r"\raid_v2_final_results.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("DETAILED RESULTS SAVED")
print("======================")

print(
    "Output:",
    OUTPUT_FILE
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("V2 RAID FINAL SUMMARY")
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
    "V2 RAID EXTERNAL "
    "EVALUATION COMPLETED"
)