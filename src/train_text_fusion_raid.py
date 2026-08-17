import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


# ============================================================
# CONFIGURATION
# ============================================================

HC3_TRAIN = r"datasets\text\HC3\train.csv"
HC3_VALIDATION = r"datasets\text\HC3\validation.csv"

RAID_TRAIN = r"datasets\text\RAID\raid_training_sample.csv"

MODEL_DIR = r"models\text_fusion_raid"


TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_TFIDF_FEATURES = 150000

BATCH_SIZE = 32

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print()
print("TEXT FUSION + RAID DETECTOR")
print("===========================")

print(
    "Transformer:",
    TRANSFORMER_NAME
)


# ============================================================
# LOAD DATA
# ============================================================

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
    "HC3 training samples:",
    len(hc3_train)
)

print(
    "HC3 validation samples:",
    len(hc3_validation)
)

print(
    "RAID training samples:",
    len(raid_train)
)


# ============================================================
# SOURCE-BALANCED HC3
# ============================================================

print()
print("BUILDING SOURCE-BALANCED HC3 TRAINING SET")
print("==========================================")


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
    "Balanced HC3 samples:",
    len(hc3_balanced)
)


# ============================================================
# PREPARE RAID COLUMNS
# ============================================================

print()
print("PREPARING RAID TRAINING DATA")
print("============================")


raid_train = raid_train[
    [
        "text",
        "label"
    ]
].copy()


raid_train["text"] = (
    raid_train["text"]
    .fillna("")
    .astype(str)
)


raid_train["label"] = (
    raid_train["label"]
    .astype(int)
)


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

print()
print("COMBINING HC3 + RAID")
print("====================")


hc3_train_final = hc3_balanced[
    [
        "text",
        "label"
    ]
].copy()


hc3_train_final["text"] = (
    hc3_train_final["text"]
    .fillna("")
    .astype(str)
)


hc3_train_final["label"] = (
    hc3_train_final["label"]
    .astype(int)
)


combined_train = pd.concat(
    [
        hc3_train_final,
        raid_train
    ],
    ignore_index=True
)


combined_train = combined_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


validation = hc3_validation[
    [
        "text",
        "label"
    ]
].copy()


validation["text"] = (
    validation["text"]
    .fillna("")
    .astype(str)
)


validation["label"] = (
    validation["label"]
    .astype(int)
)


print(
    "Final training samples:",
    len(combined_train)
)

print(
    "Validation samples:",
    len(validation)
)


print()
print("FINAL TRAIN LABEL DISTRIBUTION")
print("==============================")

print(
    combined_train["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

    text = str(text)

    words = text.split()

    word_count = len(
        words
    )

    sentence_count = max(
        text.count(".")
        + text.count("!")
        + text.count("?"),
        1
    )

    avg_sentence_length = (
        word_count /
        sentence_count
    )

    unique_words = len(
        set(
            w.lower()
            for w in words
        )
    )

    vocabulary_diversity = (
        unique_words /
        max(
            word_count,
            1
        )
    )

    return [
        word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity
    ]


print()
print("BUILDING LINGUISTIC FEATURES")
print("============================")


train_linguistic = np.array(
    [
        linguistic_features(x)
        for x in combined_train["text"]
    ],
    dtype=np.float32
)


validation_linguistic = np.array(
    [
        linguistic_features(x)
        for x in validation["text"]
    ],
    dtype=np.float32
)


print(
    "Training linguistic shape:",
    train_linguistic.shape
)

print(
    "Validation linguistic shape:",
    validation_linguistic.shape
)


# ============================================================
# SCALE LINGUISTIC FEATURES
# ============================================================

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


# ============================================================
# WORD TF-IDF
# ============================================================

print()
print("BUILDING WORD TF-IDF")
print("====================")


word_vectorizer = TfidfVectorizer(
    max_features=MAX_TFIDF_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.98
)


print(
    "Fitting word vocabulary..."
)


X_train_word = (
    word_vectorizer.fit_transform(
        combined_train["text"]
    )
)


print(
    "Transforming validation..."
)


X_validation_word = (
    word_vectorizer.transform(
        validation["text"]
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


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print()
print("BUILDING CHARACTER TF-IDF")
print("==========================")


char_vectorizer = TfidfVectorizer(
    analyzer="char",
    max_features=MAX_TFIDF_FEATURES,
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=2,
    max_df=0.98
)


print(
    "Fitting character vocabulary..."
)


X_train_char = (
    char_vectorizer.fit_transform(
        combined_train["text"]
    )
)


print(
    "Transforming validation..."
)


X_validation_char = (
    char_vectorizer.transform(
        validation["text"]
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


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("LOADING MINILM")
print("==============")


transformer = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print()
print("CREATING TRAINING EMBEDDINGS")
print("============================")


X_train_embedding = (
    transformer.encode(
        combined_train["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
)


print()
print("Creating validation embeddings...")


X_validation_embedding = (
    transformer.encode(
        validation["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
)


print(
    "Training embedding shape:",
    X_train_embedding.shape
)

print(
    "Validation embedding shape:",
    X_validation_embedding.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("COMBINING FEATURES")
print("==================")


X_train = hstack(
    [
        X_train_word,

        X_train_char,

        csr_matrix(
            X_train_embedding
        ),

        csr_matrix(
            train_linguistic
        )
    ]
).tocsr()


X_validation = hstack(
    [
        X_validation_word,

        X_validation_char,

        csr_matrix(
            X_validation_embedding
        ),

        csr_matrix(
            validation_linguistic
        )
    ]
).tocsr()


print(
    "Combined training shape:",
    X_train.shape
)

print(
    "Combined validation shape:",
    X_validation.shape
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("TRAINING FINAL FUSION CLASSIFIER")
print("===============================")


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2.0,
    solver="liblinear",
    random_state=RANDOM_STATE
)


classifier.fit(
    X_train,
    combined_train["label"]
)


print(
    "Training completed."
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("EVALUATING HC3 VALIDATION SET")
print("=============================")


y_true = validation[
    "label"
].values


y_pred = classifier.predict(
    X_validation
)


y_probability = classifier.predict_proba(
    X_validation
)[:, 1]


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_true,
    y_probability
)


print()
print("TEXT FUSION + RAID RESULTS")
print("===========================")

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


print()
print("CONFUSION MATRIX")
print("================")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


print()
print("CLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("SAVING FINAL MODEL")
print("==================")


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

    "model_type":
        "hc3_raid_word_char_minilm_linguistic",

    "transformer":
        TRANSFORMER_NAME,

    "word_features":
        MAX_TFIDF_FEATURES,

    "character_features":
        MAX_TFIDF_FEATURES,

    "embedding_dimension":
        384,

    "linguistic_features":
        4,

    "training_samples":
        len(combined_train),

    "hc3_balanced_samples":
        len(hc3_train_final),

    "raid_training_samples":
        len(raid_train),

    "validation_samples":
        len(validation),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1":
        float(f1),

    "roc_auc":
        float(roc_auc)

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

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1": float(f1),

    "roc_auc": float(roc_auc),

    "confusion_matrix":
        confusion_matrix(
            y_true,
            y_pred
        ).tolist()

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


print()
print("FINAL MODEL SAVED")
print("=================")

print(
    "Directory:",
    MODEL_DIR
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


print()
print(
    "TEXT FUSION + RAID TRAINING COMPLETED"
)