import os
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

HC3_TRAIN = "datasets\\text\\HC3\\train.csv"
HC3_VALIDATION = "datasets\\text\\HC3\\validation.csv"

RAID_TRAIN = (
    "datasets\\text\\RAID\\raid_domain_training_sample.csv"
)

MODEL_DIR = "models\\text_fusion_raid_domain"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RANDOM_STATE = 42


# =========================================================
# HEADER
# =========================================================

print()
print("TEXT FUSION + DOMAIN-BALANCED RAID DETECTOR")
print("============================================")
print(
    f"Transformer: {TRANSFORMER_NAME}"
)


# =========================================================
# LOAD DATA
# =========================================================

print()
print("LOADING DATA")
print("============")


hc3_train = pd.read_csv(
    HC3_TRAIN
)

hc3_validation = pd.read_csv(
    HC3_VALIDATION
)

raid_train = pd.read_csv(
    RAID_TRAIN
)


print(
    f"HC3 training samples: {len(hc3_train)}"
)

print(
    f"HC3 validation samples: {len(hc3_validation)}"
)

print(
    f"RAID domain training samples: {len(raid_train)}"
)


# =========================================================
# SOURCE-BALANCED HC3 TRAINING DATA
# =========================================================

print()
print(
    "BUILDING SOURCE-BALANCED HC3 TRAINING SET"
)
print(
    "=========================================="
)


balanced_parts = []


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


    n = min(
        len(human),
        len(ai)
    )


    # Keep the same maximum used by our
    # previous balanced MiniLM/fusion models.
    if source == "reddit_eli5":

        n = min(
            n,
            5000
        )


    human_sample = human.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    ai_sample = ai.sample(
        n=n,
        random_state=RANDOM_STATE
    )


    balanced_parts.append(
        human_sample
    )

    balanced_parts.append(
        ai_sample
    )


    print(
        f"{source:12s} "
        f"Human={len(human_sample):5d} "
        f"AI={len(ai_sample):5d}"
    )


hc3_balanced = pd.concat(
    balanced_parts,
    ignore_index=True
)


print()
print(
    f"Balanced HC3 samples: "
    f"{len(hc3_balanced)}"
)


# =========================================================
# PREPARE RAID TRAINING DATA
# =========================================================

print()
print(
    "PREPARING DOMAIN-BALANCED RAID DATA"
)
print(
    "===================================="
)


raid_train = raid_train[
    [
        "text",
        "raid_model",
        "domain",
        "label",
        "label_name",
    ]
].copy()


raid_train["source"] = (
    "raid_" +
    raid_train["domain"].astype(str)
)


# =========================================================
# PREPARE HC3 TEXT
# =========================================================

hc3_balanced = hc3_balanced.copy()

hc3_balanced["text"] = (
    hc3_balanced["text"]
    .fillna("")
    .astype(str)
)


hc3_validation = hc3_validation.copy()

hc3_validation["text"] = (
    hc3_validation["text"]
    .fillna("")
    .astype(str)
)


# =========================================================
# COMBINE HC3 + RAID
# =========================================================

print()
print("COMBINING HC3 + RAID")
print("====================")


training_df = pd.concat(
    [
        hc3_balanced[
            ["text", "label"]
        ],
        raid_train[
            ["text", "label"]
        ],
    ],
    ignore_index=True
)


validation_df = hc3_validation[
    ["text", "label"]
].copy()


training_texts = (
    training_df["text"]
    .tolist()
)

validation_texts = (
    validation_df["text"]
    .tolist()
)


y_train = (
    training_df["label"]
    .astype(int)
    .values
)

y_validation = (
    validation_df["label"]
    .astype(int)
    .values
)


print(
    f"Final training samples: "
    f"{len(training_df)}"
)

print(
    f"Validation samples: "
    f"{len(validation_df)}"
)


print()
print(
    "FINAL TRAIN LABEL DISTRIBUTION"
)
print(
    "=============================="
)

print(
    training_df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# LINGUISTIC FEATURES
# =========================================================

def linguistic_features(texts):

    features = []


    for text in texts:

        text = str(text)


        words = text.split()

        word_count = len(words)


        sentences = [
            s.strip()
            for s in text.replace(
                "!",
                "."
            ).replace(
                "?",
                "."
            ).split(".")
            if s.strip()
        ]


        sentence_count = max(
            1,
            len(sentences)
        )


        avg_sentence_length = (
            word_count /
            sentence_count
        )


        if word_count > 0:

            vocabulary_diversity = (
                len(set(
                    w.lower()
                    for w in words
                ))
                /
                word_count
            )

        else:

            vocabulary_diversity = 0.0


        avg_word_length = (
            sum(
                len(w)
                for w in words
            )
            /
            max(
                1,
                word_count
            )
        )


        features.append(
            [
                word_count,
                sentence_count,
                avg_sentence_length,
                vocabulary_diversity,
                avg_word_length,
            ]
        )


    return np.asarray(
        features,
        dtype=np.float32
    )


print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


X_ling_train_raw = linguistic_features(
    training_texts
)

X_ling_validation_raw = linguistic_features(
    validation_texts
)


print(
    "Training linguistic shape:",
    X_ling_train_raw.shape
)

print(
    "Validation linguistic shape:",
    X_ling_validation_raw.shape
)


# =========================================================
# SCALE LINGUISTIC FEATURES
# =========================================================

linguistic_scaler = StandardScaler()


X_ling_train = (
    linguistic_scaler.fit_transform(
        X_ling_train_raw
    )
)


X_ling_validation = (
    linguistic_scaler.transform(
        X_ling_validation_raw
    )
)


# =========================================================
# WORD TF-IDF
# =========================================================

print()
print("BUILDING WORD TF-IDF")
print("====================")


word_vectorizer = TfidfVectorizer(
    max_features=150000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
)


print(
    "Fitting word vocabulary..."
)


X_word_train = (
    word_vectorizer.fit_transform(
        training_texts
    )
)


print(
    "Transforming validation..."
)


X_word_validation = (
    word_vectorizer.transform(
        validation_texts
    )
)


print(
    "Training word shape:",
    X_word_train.shape
)

print(
    "Validation word shape:",
    X_word_validation.shape
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

print()
print("BUILDING CHARACTER TF-IDF")
print("=========================")


char_vectorizer = TfidfVectorizer(
    analyzer="char",
    max_features=150000,
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
)


print(
    "Fitting character vocabulary..."
)


X_char_train = (
    char_vectorizer.fit_transform(
        training_texts
    )
)


print(
    "Transforming validation..."
)


X_char_validation = (
    char_vectorizer.transform(
        validation_texts
    )
)


print(
    "Training character shape:",
    X_char_train.shape
)

print(
    "Validation character shape:",
    X_char_validation.shape
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
# CREATE EMBEDDINGS
# =========================================================

print()
print("CREATING TRAINING EMBEDDINGS")
print("============================")


X_embed_train = transformer.encode(
    training_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


print(
    "Training embedding shape:",
    X_embed_train.shape
)


print()
print("CREATING VALIDATION EMBEDDINGS")
print("================================")


X_embed_validation = transformer.encode(
    validation_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


print(
    "Validation embedding shape:",
    X_embed_validation.shape
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("COMBINING FEATURES")
print("==================")


X_train = hstack(
    [
        X_word_train,
        X_char_train,
        X_embed_train,
        X_ling_train,
    ],
    format="csr"
)


X_validation = hstack(
    [
        X_word_validation,
        X_char_validation,
        X_embed_validation,
        X_ling_validation,
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
# TRAIN CLASSIFIER
# =========================================================

print()
print(
    "TRAINING FINAL DOMAIN-AWARE FUSION CLASSIFIER"
)
print(
    "=============================================="
)


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    random_state=RANDOM_STATE,
)


classifier.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# =========================================================
# VALIDATION
# =========================================================

print()
print("EVALUATING HC3 VALIDATION SET")
print("=============================")


predictions = classifier.predict(
    X_validation
)

probabilities = classifier.predict_proba(
    X_validation
)[:, 1]


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

roc_auc = roc_auc_score(
    y_validation,
    probabilities
)


print()
print(
    "DOMAIN-AWARE FUSION MODEL RESULTS"
)
print(
    "================================="
)

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


cm = confusion_matrix(
    y_validation,
    predictions
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
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

print()
print("SAVING DOMAIN-AWARE MODEL")
print("==========================")


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


with open(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        word_vectorizer,
        f
    )


with open(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        char_vectorizer,
        f
    )


with open(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        linguistic_scaler,
        f
    )


with open(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        classifier,
        f
    )


config = {
    "transformer": TRANSFORMER_NAME,
    "model_type": (
        "hc3_raid_domain_balanced_fusion"
    ),
    "word_features": 150000,
    "char_features": 150000,
    "embedding_features": 384,
    "linguistic_features": 5,
    "total_features": X_train.shape[1],
    "raid_training_samples": len(raid_train),
    "hc3_balanced_samples": len(
        hc3_balanced
    ),
    "validation_samples": len(
        validation_df
    ),
}


with open(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        config,
        f
    )


metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
}


with open(
    os.path.join(
        MODEL_DIR,
        "text_metrics.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        metrics,
        f
    )


# Save training distribution
training_distribution = (
    training_df["label"]
    .value_counts()
    .sort_index()
    .rename_axis("label")
    .reset_index(
        name="count"
    )
)


training_distribution.to_csv(
    os.path.join(
        MODEL_DIR,
        "training_distribution.csv"
    ),
    index=False
)


# =========================================================
# FINAL
# =========================================================

print()
print("DOMAIN-AWARE MODEL SAVED")
print("========================")


print(
    f"Directory: {MODEL_DIR}"
)


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


print()
print(
    "TEXT FUSION + DOMAIN-BALANCED RAID "
    "TRAINING COMPLETED"
)