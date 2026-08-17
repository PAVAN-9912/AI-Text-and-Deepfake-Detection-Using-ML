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
    confusion_matrix
)

from sentence_transformers import SentenceTransformer


# =========================================================
# SETTINGS
# =========================================================

MODEL_DIR = r"models\text_final_v2"

HC3_FILE = (
    r"datasets\text\HC3\validation.csv"
)

SENTENCE_FILE = (
    r"datasets\text\SentenceAI"
    r"\sentence_ai_v2_test.csv"
)

RAID_FILE = (
    r"datasets\text\RAID"
    r"\raid_external.csv"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

THRESHOLDS = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75
]


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("V2 EXTERNAL THRESHOLD COMPARISON")
print("=" * 60)
print()


# =========================================================
# LINGUISTIC FEATURES
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
# FIND SENTENCEAI FILE
# =========================================================

def find_sentence_file():

    candidates = [
        SENTENCE_FILE,
        r"datasets\text\SentenceAI\sentence_ai_test.csv",
        r"datasets\text\SentenceAI\sentence_ai_training.csv",
    ]

    for path in candidates:

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "SentenceAI test file not found."
    )


# =========================================================
# STANDARDIZE DATASET
# =========================================================

def standardize_dataframe(
    df,
    dataset_name
):

    df = df.copy()

    # -----------------------------------------------------
    # TEXT COLUMN
    # -----------------------------------------------------

    if "text" in df.columns:

        text_column = "text"

    elif "generation" in df.columns:

        text_column = "generation"

    else:

        raise ValueError(
            f"No text column found in {dataset_name}"
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


    # -----------------------------------------------------
    # LABEL COLUMN
    # -----------------------------------------------------

    if "label" in df.columns:

        labels = pd.to_numeric(
            df["label"],
            errors="coerce"
        )

    elif "label_name" in df.columns:

        labels = (
            df["label_name"]
            .astype(str)
            .str.lower()
            .map({
                "human": 0,
                "ai": 1
            })
        )

    else:

        raise ValueError(
            f"No label column found in {dataset_name}"
        )

    df["label"] = labels

    df = df[
        df["label"].isin([0, 1])
    ].copy()

    df["label"] = (
        df["label"]
        .astype(int)
    )


    # -----------------------------------------------------
    # LENGTH
    # -----------------------------------------------------

    def get_length(text):

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
        .apply(get_length)
    )

    df["dataset"] = dataset_name

    return df[
        [
            "text",
            "label",
            "length_group",
            "dataset"
        ]
    ].copy()


# =========================================================
# LOAD DATA
# =========================================================

print("LOADING EXTERNAL DATA")
print("=====================")


# ---------------------------------------------------------
# HC3
# ---------------------------------------------------------

print()
print("HC3")

hc3 = pd.read_csv(
    HC3_FILE
)

hc3 = standardize_dataframe(
    hc3,
    "HC3"
)

print(
    f"Samples: {len(hc3)}"
)


# ---------------------------------------------------------
# SentenceAI
# ---------------------------------------------------------

print()
print("SentenceAI")

sentence_file = (
    find_sentence_file()
)

print(
    f"File: {sentence_file}"
)

sentence = pd.read_csv(
    sentence_file
)

sentence = standardize_dataframe(
    sentence,
    "SentenceAI"
)

print(
    f"Samples: {len(sentence)}"
)


# ---------------------------------------------------------
# RAID
# ---------------------------------------------------------

print()
print("RAID")

raid = pd.read_csv(
    RAID_FILE
)

raid = standardize_dataframe(
    raid,
    "RAID"
)

print(
    f"Samples: {len(raid)}"
)


# =========================================================
# IMPORTANT
# =========================================================

# We use the existing external datasets exactly as they
# are. No training data is modified here.


datasets = [
    hc3,
    sentence,
    raid
]


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
# LOAD MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")

transformer = SentenceTransformer(
    TRANSFORMER_NAME
)

print(
    "MiniLM loaded successfully."
)


# =========================================================
# PROCESS EACH DATASET
# =========================================================

all_results = []


for dataset in datasets:

    dataset_name = (
        dataset["dataset"].iloc[0]
    )

    print()
    print("=" * 60)
    print(
        f"PROCESSING {dataset_name}"
    )
    print("=" * 60)

    texts = (
        dataset["text"]
        .tolist()
    )

    y_true = (
        dataset["label"]
        .to_numpy()
    )


    # -----------------------------------------------------
    # WORD
    # -----------------------------------------------------

    print()
    print("Building word features...")

    word_features = (
        word_vectorizer.transform(
            texts
        )
    )


    # -----------------------------------------------------
    # CHARACTER
    # -----------------------------------------------------

    print(
        "Building character features..."
    )

    char_features = (
        char_vectorizer.transform(
            texts
        )
    )


    # -----------------------------------------------------
    # MINILM
    # -----------------------------------------------------

    print(
        "Creating MiniLM embeddings..."
    )

    embeddings = transformer.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embedding_features = csr_matrix(
        embeddings
    )


    # -----------------------------------------------------
    # LINGUISTIC
    # -----------------------------------------------------

    print(
        "Building linguistic features..."
    )

    linguistic = (
        build_linguistic_features(
            texts
        )
    )

    linguistic = (
        linguistic_scaler.transform(
            linguistic
        )
    )

    linguistic_features = csr_matrix(
        linguistic
    )


    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------

    combined = hstack(
        [
            word_features,
            char_features,
            embedding_features,
            linguistic_features
        ],
        format="csr"
    )


    # -----------------------------------------------------
    # PROBABILITIES
    # -----------------------------------------------------

    probabilities = (
        classifier.predict_proba(
            combined
        )[:, 1]
    )


    # -----------------------------------------------------
    # THRESHOLD RESULTS
    # -----------------------------------------------------

    print()
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Human FP':<12}"
        f"{'AI FN':<12}"
    )

    print("-" * 72)


    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{fp:<12}"
            f"{fn:<12}"
        )

        all_results.append(
            {
                "dataset": dataset_name,
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "human_false_positives": int(fp),
                "ai_false_negatives": int(fn),
                "samples": len(y_true)
            }
        )


# =========================================================
# COMBINED RESULTS
# =========================================================

results = pd.DataFrame(
    all_results
)


# =========================================================
# COMBINED WEIGHTED METRICS
# =========================================================

print()
print("=" * 60)
print("COMBINED EXTERNAL THRESHOLD RESULTS")
print("=" * 60)

print()
print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 60)


combined_rows = []


for threshold in THRESHOLDS:

    threshold_results = (
        results[
            results["threshold"]
            == threshold
        ]
    )

    total_samples = (
        threshold_results["samples"]
        .sum()
    )


    # Calculate totals from FP/FN.
    total_human = 0
    total_ai = 0
    total_fp = (
        threshold_results[
            "human_false_positives"
        ].sum()
    )
    total_fn = (
        threshold_results[
            "ai_false_negatives"
        ].sum()
    )


    for dataset in datasets:

        name = (
            dataset["dataset"].iloc[0]
        )

        total_human += (
            dataset["label"]
            .eq(0)
            .sum()
        )

        total_ai += (
            dataset["label"]
            .eq(1)
            .sum()
        )


    total_correct = (
        total_human
        -
        total_fp
        +
        total_ai
        -
        total_fn
    )

    total_accuracy = (
        total_correct /
        total_samples
    )

    total_tp = (
        total_ai -
        total_fn
    )

    total_precision = (
        total_tp /
        max(
            total_tp + total_fp,
            1
        )
    )

    total_recall = (
        total_tp /
        max(total_ai, 1)
    )

    total_f1 = (
        2 *
        total_precision *
        total_recall
        /
        max(
            total_precision +
            total_recall,
            1e-12
        )
    )


    print(
        f"{threshold:<12.2f}"
        f"{total_accuracy:<12.4f}"
        f"{total_precision:<12.4f}"
        f"{total_recall:<12.4f}"
        f"{total_f1:<12.4f}"
    )


    combined_rows.append(
        {
            "threshold": threshold,
            "accuracy": total_accuracy,
            "precision": total_precision,
            "recall": total_recall,
            "f1": total_f1,
            "human_false_positives": int(
                total_fp
            ),
            "ai_false_negatives": int(
                total_fn
            ),
            "samples": int(
                total_samples
            )
        }
    )


combined = pd.DataFrame(
    combined_rows
)


# =========================================================
# BEST BALANCED THRESHOLD
# =========================================================

best = (
    combined
    .sort_values(
        "f1",
        ascending=False
    )
    .iloc[0]
)


print()
print("=" * 60)
print("BEST COMBINED THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best['threshold']:.2f}"
)

print(
    f"Accuracy  : {best['accuracy']:.4f}"
)

print(
    f"Precision : {best['precision']:.4f}"
)

print(
    f"Recall    : {best['recall']:.4f}"
)

print(
    f"F1        : {best['f1']:.4f}"
)

print(
    f"Human FP  : {int(best['human_false_positives'])}"
)

print(
    f"AI FN     : {int(best['ai_false_negatives'])}"
)


# =========================================================
# SAVE
# =========================================================

output_file = (
    r"datasets\text\v2_external_threshold_comparison.csv"
)

results.to_csv(
    output_file,
    index=False
)


combined_file = (
    r"datasets\text\v2_external_threshold_combined.csv"
)

combined.to_csv(
    combined_file,
    index=False
)


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 60)
print("EXTERNAL THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)

print()
print(
    "Detailed output:",
    output_file
)

print(
    "Combined output:",
    combined_file
)

print()