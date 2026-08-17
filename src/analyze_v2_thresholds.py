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

TEST_FILE = r"datasets\text\realworld_test.csv"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("V2 DECISION THRESHOLD ANALYSIS")
print("=" * 60)
print()


# =========================================================
# LOAD DATA
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

y_true = (
    df["label"]
    .astype(int)
    .to_numpy()
)

texts = (
    df["text"]
    .tolist()
)

print(
    f"Samples: {len(df)}"
)


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
# LOAD V2 COMPONENTS
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


print("V2 model loaded.")


# =========================================================
# WORD FEATURES
# =========================================================

print()
print("BUILDING WORD FEATURES")
print("======================")

word_features = (
    word_vectorizer.transform(
        texts
    )
)

print(
    "Shape:",
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
        texts
    )
)

print(
    "Shape:",
    char_features.shape
)


# =========================================================
# MINILM
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
# EMBEDDINGS
# =========================================================

print()
print("CREATING EMBEDDINGS")
print("===================")

embeddings = transformer.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    "Shape:",
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
        texts
    )
)

linguistic_features = (
    linguistic_scaler.transform(
        linguistic_features
    )
)

linguistic_features = csr_matrix(
    linguistic_features
)

print(
    "Shape:",
    linguistic_features.shape
)


# =========================================================
# COMBINE
# =========================================================

print()
print("COMBINING FEATURES")
print("==================")

features = hstack(
    [
        word_features,
        char_features,
        embedding_features,
        linguistic_features
    ],
    format="csr"
)

print(
    "Combined shape:",
    features.shape
)


# =========================================================
# AI PROBABILITIES
# =========================================================

print()
print("CALCULATING AI PROBABILITIES")
print("============================")

ai_probabilities = (
    classifier.predict_proba(
        features
    )[:, 1]
)


# =========================================================
# THRESHOLD TESTING
# =========================================================

print()
print("=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

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


threshold_results = []


for threshold in np.arange(
    0.30,
    0.76,
    0.05
):

    predictions = (
        ai_probabilities >= threshold
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
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    threshold_results.append(
        {
            "threshold": round(
                float(threshold),
                2
            ),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "human_false_positives": int(fp),
            "ai_false_negatives": int(fn)
        }
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


# =========================================================
# BEST F1
# =========================================================

results_df = pd.DataFrame(
    threshold_results
)

best_f1_row = (
    results_df
    .sort_values(
        "f1",
        ascending=False
    )
    .iloc[0]
)


print()
print("=" * 60)
print("BEST THRESHOLD BY F1")
print("=" * 60)

print(
    f"Threshold : "
    f"{best_f1_row['threshold']:.2f}"
)

print(
    f"Accuracy  : "
    f"{best_f1_row['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{best_f1_row['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_f1_row['recall']:.4f}"
)

print(
    f"F1        : "
    f"{best_f1_row['f1']:.4f}"
)


# =========================================================
# HUMAN-FRIENDLY THRESHOLD ANALYSIS
# =========================================================

print()
print("=" * 60)
print("HUMAN FALSE-POSITIVE ANALYSIS")
print("=" * 60)


for threshold in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]:

    predictions = (
        ai_probabilities >= threshold
    ).astype(int)

    human_mask = (
        y_true == 0
    )

    human_false_positive = (
        predictions[human_mask] == 1
    ).sum()

    human_total = (
        human_mask.sum()
    )

    human_accuracy = (
        human_total -
        human_false_positive
    ) / human_total

    ai_mask = (
        y_true == 1
    )

    ai_correct = (
        predictions[ai_mask] == 1
    ).sum()

    ai_total = (
        ai_mask.sum()
    )

    ai_recall = (
        ai_correct /
        ai_total
    )

    print()

    print(
        f"Threshold {threshold:.2f}"
    )

    print(
        f"Human correctly identified: "
        f"{human_total - human_false_positive}/"
        f"{human_total}"
    )

    print(
        f"Human accuracy: "
        f"{human_accuracy:.4f}"
    )

    print(
        f"AI correctly identified: "
        f"{ai_correct}/"
        f"{ai_total}"
    )

    print(
        f"AI recall: "
        f"{ai_recall:.4f}"
    )


# =========================================================
# INDIVIDUAL PROBABILITIES
# =========================================================

print()
print("=" * 60)
print("INDIVIDUAL AI PROBABILITIES")
print("=" * 60)


for i in range(
    len(df)
):

    actual = (
        "AI"
        if y_true[i] == 1
        else "Human"
    )

    probability = (
        ai_probabilities[i]
    )

    print()
    print(
        f"{i + 1:02d}. "
        f"Actual={actual:<5} "
        f"AI={probability * 100:6.2f}% "
        f"| "
        f"{df.iloc[i]['category']}"
    )

    print(
        df.iloc[i]["text"]
    )


# =========================================================
# SAVE
# =========================================================

output_file = (
    r"datasets\text\realworld_threshold_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()
print("=" * 60)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)

print()
print(
    "Output:",
    output_file
)

print()