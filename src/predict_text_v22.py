import os
import pickle
import numpy as np

from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer


# ============================================================
# AI TEXT DETECTOR V2.2
# ADAPTIVE INFERENCE
# ============================================================

print()
print("=" * 60)
print("GENERALIZED AI TEXT DETECTOR V2.2")
print("=" * 60)
print("V2 IMPROVED + LENGTH-AWARE INFERENCE")
print()


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = r"models\text_final_v2_improved"

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# Length-aware thresholds
#
# These are deliberately conservative.
#
# We are NOT retraining the model.
# We are only changing the final decision rule.
# ------------------------------------------------------------

THRESHOLDS = {

    "very_short": 0.45,

    "short": 0.47,

    "medium": 0.50,

    "long": 0.47
}


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

print("LOADING V2 IMPROVED MODEL")
print("===========================")


def load_pickle(filename):

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    with open(path, "rb") as f:

        return pickle.load(f)


word_vectorizer = load_pickle(
    "word_tfidf_vectorizer.pkl"
)

char_vectorizer = load_pickle(
    "char_tfidf_vectorizer.pkl"
)

linguistic_scaler = load_pickle(
    "linguistic_scaler.pkl"
)

classifier = load_pickle(
    "text_classifier.pkl"
)

config = load_pickle(
    "text_config.pkl"
)


print(
    "Model components loaded successfully."
)


print(
    "Expected feature count:",
    config["feature_count"]
)


# ============================================================
# LOAD MINILM
# ============================================================

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


# ============================================================
# LENGTH GROUP
# ============================================================

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


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

def build_linguistic_features(texts):

    features = []

    for text in texts:

        text = str(text)

        words = text.split()

        word_count = len(words)


        # ----------------------------------------------------
        # Sentence count
        # ----------------------------------------------------

        sentence_count = sum(
            1
            for char in text
            if char in ".!?"
        )

        if sentence_count == 0:

            sentence_count = 1


        # ----------------------------------------------------
        # Average sentence length
        # ----------------------------------------------------

        avg_sentence_length = (
            word_count /
            max(sentence_count, 1)
        )


        # ----------------------------------------------------
        # Vocabulary diversity
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Log word count
        # ----------------------------------------------------

        log_word_count = np.log1p(
            word_count
        )


        # ----------------------------------------------------
        # Length bucket
        # ----------------------------------------------------

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


# ============================================================
# GET USER TEXT
# ============================================================

print()
print("=" * 60)
print("TEXT INPUT")
print("=" * 60)

print()
print("Enter the text you want to analyze.")
print("You can enter multiple sentences.")
print("Type END on a new line when finished.")
print()


lines = []


while True:

    line = input()

    if line.strip().upper() == "END":

        break

    lines.append(line)


text = "\n".join(lines).strip()


if not text:

    raise ValueError(
        "No text was entered."
    )


# ============================================================
# BASIC INFORMATION
# ============================================================

word_count = len(
    text.split()
)

length_group = get_length_group(
    text
)


threshold = THRESHOLDS[
    length_group
]


print()
print("=" * 60)
print("TEXT ANALYSIS")
print("=" * 60)


print(
    "Word count     :",
    word_count
)

print(
    "Length group   :",
    length_group
)

print(
    "Decision threshold:",
    f"{threshold:.2f}"
)


# ============================================================
# WORD TF-IDF
# ============================================================

print()
print("BUILDING WORD FEATURES")
print("=======================")


X_word = (
    word_vectorizer.transform(
        [text]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        [text]
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# ============================================================
# MINILM EMBEDDING
# ============================================================

print()
print("CREATING MINILM EMBEDDING")
print("==========================")


X_embedding = embedder.encode(
    [text],
    batch_size=1,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    X_embedding.shape
)


# ============================================================
# LINGUISTIC FEATURES
# ============================================================

print()
print("BUILDING LINGUISTIC FEATURES")
print("=============================")


X_linguistic_raw = (
    build_linguistic_features(
        [text]
    )
)


X_linguistic = (
    linguistic_scaler.transform(
        X_linguistic_raw
    )
)


print(
    "Linguistic shape:",
    X_linguistic.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_embedding_sparse = csr_matrix(
    X_embedding
)

X_linguistic_sparse = csr_matrix(
    X_linguistic
)


X = hstack(
    [
        X_word,
        X_char,
        X_embedding_sparse,
        X_linguistic_sparse
    ],
    format="csr"
)


print()
print("COMBINED FEATURES")
print("==================")


print(
    "Feature shape:",
    X.shape
)


expected_features = (
    config["feature_count"]
)

actual_features = (
    X.shape[1]
)


print(
    "Expected:",
    expected_features
)

print(
    "Actual  :",
    actual_features
)


if actual_features != expected_features:

    raise RuntimeError(
        "Feature count mismatch."
    )


# ============================================================
# PREDICTION
# ============================================================

print()
print("RUNNING V2.2 PREDICTION")
print("========================")


probability = float(
    classifier.predict_proba(
        X
    )[0, 1]
)


prediction = (
    probability >= threshold
)


# ============================================================
# CONFIDENCE
# ============================================================

if prediction:

    confidence = probability

    label = "AI-GENERATED"

else:

    confidence = 1.0 - probability

    label = "HUMAN-WRITTEN"


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("V2.2 FINAL RESULT")
print("=" * 60)


print()
print(
    "Prediction       :",
    label
)

print(
    "AI Probability   :",
    f"{probability * 100:.2f}%"
)

print(
    "Human Probability:",
    f"{(1 - probability) * 100:.2f}%"
)

print(
    "Confidence       :",
    f"{confidence * 100:.2f}%"
)

print(
    "Length Group     :",
    length_group
)

print(
    "Threshold Used   :",
    f"{threshold:.2f}"
)


print()
print("=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)