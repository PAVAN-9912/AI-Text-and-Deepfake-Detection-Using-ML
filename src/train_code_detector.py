import os
import re
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import hstack

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
    classification_report,
)

from sentence_transformers import SentenceTransformer


print()
print("CODE-AI SPECIALIST DETECTOR")
print("===========================")
print("AIGCodeSet")
print()


# =========================================================
# SETTINGS
# =========================================================

SEED = 42

TRAIN_PATH = (
    r"datasets\text\CodeAI\aigcodeset_training.csv"
)

TEST_PATH = (
    r"datasets\text\CodeAI\aigcodeset_test.csv"
)

MODEL_DIR = (
    r"models\code_detector"
)

TRANSFORMER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

WORD_MAX_FEATURES = 100000
CHAR_MAX_FEATURES = 100000

BATCH_SIZE = 32


# =========================================================
# DIRECTORIES
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

print()
print("LOADING CODE DATA")
print("=================")


train_df = pd.read_csv(
    TRAIN_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)


print(
    f"Training samples: {len(train_df)}"
)

print(
    f"Test samples    : {len(test_df)}"
)


# =========================================================
# BASIC VALIDATION
# =========================================================

required_columns = [
    "text",
    "label",
    "label_name",
    "generator",
    "length_group",
]


for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing training column: {column}"
        )

    if column not in test_df.columns:

        raise ValueError(
            f"Missing test column: {column}"
        )


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print()
print("TRAIN LABEL DISTRIBUTION")
print("=========================")

print(
    train_df[
        "label_name"
    ].value_counts().to_string()
)


print()
print("TEST LABEL DISTRIBUTION")
print("=======================")

print(
    test_df[
        "label_name"
    ].value_counts().to_string()
)


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("TRAIN LENGTH DISTRIBUTION")
print("==========================")

print(
    pd.crosstab(
        train_df["length_group"],
        train_df["label_name"]
    ).to_string()
)


print()
print("TEST LENGTH DISTRIBUTION")
print("========================")

print(
    pd.crosstab(
        test_df["length_group"],
        test_df["label_name"]
    ).to_string()
)


# =========================================================
# GENERATOR DISTRIBUTION
# =========================================================

print()
print("TRAIN AI GENERATOR DISTRIBUTION")
print("===============================")

print(
    train_df[
        train_df["label"] == 1
    ]["generator"]
    .value_counts()
    .to_string()
)


print()
print("TEST AI GENERATOR DISTRIBUTION")
print("==============================")

print(
    test_df[
        test_df["label"] == 1
    ]["generator"]
    .value_counts()
    .to_string()
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_code(text):

    if pd.isna(text):

        return ""

    text = str(text)

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize tabs.
    text = text.replace(
        "\t",
        " "
    )

    # Keep code structure but remove excessive whitespace.
    text = re.sub(
        r"[ ]{4,}",
        " ",
        text
    )

    return text.strip()


print()
print("CLEANING CODE")
print("=============")


train_texts = (
    train_df["text"]
    .fillna("")
    .astype(str)
    .map(clean_code)
    .tolist()
)

test_texts = (
    test_df["text"]
    .fillna("")
    .astype(str)
    .map(clean_code)
    .tolist()
)


y_train = train_df[
    "label"
].astype(int).values

y_test = test_df[
    "label"
].astype(int).values


# =========================================================
# WORD TF-IDF
# =========================================================

print()
print("BUILDING WORD TF-IDF")
print("====================")

print(
    "Fitting word vocabulary..."
)


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=WORD_MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    max_df=0.98,
    strip_accents="unicode",
)


X_train_word = (
    word_vectorizer.fit_transform(
        train_texts
    )
)


print(
    "Transforming test..."
)


X_test_word = (
    word_vectorizer.transform(
        test_texts
    )
)


print(
    f"Training word shape: "
    f"{X_train_word.shape}"
)

print(
    f"Test word shape: "
    f"{X_test_word.shape}"
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

print()
print("BUILDING CHARACTER TF-IDF")
print("=========================")

print(
    "Fitting character vocabulary..."
)


char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 6),
    max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    max_df=0.99,
)


X_train_char = (
    char_vectorizer.fit_transform(
        train_texts
    )
)


print(
    "Transforming test..."
)


X_test_char = (
    char_vectorizer.transform(
        test_texts
    )
)


print(
    f"Training character shape: "
    f"{X_train_char.shape}"
)

print(
    f"Test character shape: "
    f"{X_test_char.shape}"
)


# =========================================================
# CODE LINGUISTIC / STRUCTURAL FEATURES
# =========================================================

def code_features(text):

    text = str(text)

    lines = text.splitlines()

    if len(lines) == 0:

        lines = [text]


    words = re.findall(
        r"\b\w+\b",
        text
    )


    word_count = len(words)

    unique_words = len(
        set(
            w.lower()
            for w in words
        )
    )


    vocabulary_diversity = (
        unique_words / word_count
        if word_count > 0
        else 0.0
    )


    char_count = len(text)


    non_empty_lines = [
        line
        for line in lines
        if line.strip()
    ]


    line_count = len(
        non_empty_lines
    )


    avg_line_length = (
        np.mean(
            [
                len(line)
                for line in non_empty_lines
            ]
        )
        if line_count > 0
        else 0.0
    )


    # -----------------------------------------------------
    # Code-specific structural indicators
    # -----------------------------------------------------

    comment_lines = 0

    blank_lines = 0

    indentation_lines = 0


    for line in lines:

        stripped = line.strip()


        if not stripped:

            blank_lines += 1


        if (
            line.startswith(" ")
            or line.startswith("\t")
        ):

            indentation_lines += 1


        if (
            stripped.startswith("#")
            or stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
        ):

            comment_lines += 1


    comment_ratio = (
        comment_lines / line_count
        if line_count > 0
        else 0.0
    )


    indentation_ratio = (
        indentation_lines / line_count
        if line_count > 0
        else 0.0
    )


    blank_ratio = (
        blank_lines / len(lines)
        if len(lines) > 0
        else 0.0
    )


    # -----------------------------------------------------
    # Common programming tokens
    # -----------------------------------------------------

    token_patterns = {

        "braces": r"[{}]",

        "parentheses": r"[()]",

        "brackets": r"[\[\]]",

        "semicolon": r";",

        "colon": r":",

        "operators": (
            r"==|!=|<=|>=|"
            r"\+=|-=|\*=|/=|"
            r"&&|\|\||"
            r"=>|->"
        ),

        "strings": (
            r"\"[^\"]*\"|'[^']*'"
        ),

        "numbers": (
            r"\b\d+(?:\.\d+)?\b"
        ),

        "function_defs": (
            r"\bdef\s+\w+|"
            r"\bfunction\s+\w+|"
            r"\bfunc\s+\w+"
        ),

        "class_defs": (
            r"\bclass\s+\w+"
        ),

        "imports": (
            r"\bimport\b|"
            r"\bfrom\s+\w+\s+import\b|"
            r"\busing\s+\w+|"
            r"#include"
        ),

        "loops": (
            r"\bfor\b|\bwhile\b"
        ),

        "conditionals": (
            r"\bif\b|\belif\b|\belse\b|"
            r"\bswitch\b|\bcase\b"
        ),

        "returns": (
            r"\breturn\b"
        ),
    }


    token_counts = []


    for pattern in token_patterns.values():

        token_counts.append(
            len(
                re.findall(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                )
            )
        )


    # -----------------------------------------------------
    # Normalized structural features
    # -----------------------------------------------------

    denominator = max(
        word_count,
        1
    )


    token_counts_normalized = [
        count / denominator
        for count in token_counts
    ]


    features = [

        # Basic size
        np.log1p(
            word_count
        ),

        np.log1p(
            char_count
        ),

        np.log1p(
            line_count
        ),

        vocabulary_diversity,

        avg_line_length / 100.0,

        # Layout
        comment_ratio,

        indentation_ratio,

        blank_ratio,

    ]


    features.extend(
        token_counts_normalized
    )


    return features


print()
print("BUILDING CODE STRUCTURAL FEATURES")
print("=================================")


train_structural = np.array(
    [
        code_features(text)
        for text in train_texts
    ],
    dtype=np.float32
)


test_structural = np.array(
    [
        code_features(text)
        for text in test_texts
    ],
    dtype=np.float32
)


print(
    "Training structural shape:",
    train_structural.shape
)

print(
    "Test structural shape:",
    test_structural.shape
)


# =========================================================
# SCALE STRUCTURAL FEATURES
# =========================================================

print()
print("SCALING STRUCTURAL FEATURES")
print("===========================")


structural_scaler = StandardScaler()


X_train_structural = (
    structural_scaler.fit_transform(
        train_structural
    )
)


X_test_structural = (
    structural_scaler.transform(
        test_structural
    )
)


# =========================================================
# LOAD MINILM
# =========================================================

print()
print("LOADING MINILM")
print("==============")


print(
    f"Transformer: {TRANSFORMER_NAME}"
)


model = SentenceTransformer(
    TRANSFORMER_NAME
)


print(
    "MiniLM loaded successfully."
)


# =========================================================
# EMBEDDINGS
# =========================================================

print()
print("CREATING TRAINING EMBEDDINGS")
print("============================")


X_train_embedding = (
    model.encode(
        train_texts,
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


print()
print("CREATING TEST EMBEDDINGS")
print("========================")


X_test_embedding = (
    model.encode(
        test_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
)


print(
    "Test embedding shape:",
    X_test_embedding.shape
)


# =========================================================
# COMBINE FEATURES
# =========================================================

print()
print("COMBINING CODE FEATURES")
print("=======================")


X_train_embedding_sparse = (
    X_train_embedding.astype(
        np.float32
    )
)

X_test_embedding_sparse = (
    X_test_embedding.astype(
        np.float32
    )
)


X_train_structural_sparse = (
    X_train_structural.astype(
        np.float32
    )
)

X_test_structural_sparse = (
    X_test_structural.astype(
        np.float32
    )
)


X_train = hstack(
    [
        X_train_word,
        X_train_char,
        X_train_embedding_sparse,
        X_train_structural_sparse,
    ],
    format="csr"
)


X_test = hstack(
    [
        X_test_word,
        X_test_char,
        X_test_embedding_sparse,
        X_test_structural_sparse,
    ],
    format="csr"
)


print(
    "Combined training shape:",
    X_train.shape
)

print(
    "Combined test shape:",
    X_test.shape
)


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print()
print("TRAINING CODE FUSION CLASSIFIER")
print("================================")


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    C=2.0,
    random_state=SEED,
)


classifier.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("RUNNING TEST PREDICTIONS")
print("=========================")


y_pred = classifier.predict(
    X_test
)


y_prob = (
    classifier.predict_proba(
        X_test
    )[:, 1]
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# =========================================================
# RESULTS
# =========================================================

print()
print("CODE-AI SPECIALIST RESULTS")
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
        y_test,
        y_pred
    )
)


print()
print("CLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# =========================================================
# RESULTS BY GENERATOR
# =========================================================

print()
print("PERFORMANCE BY AI GENERATOR")
print("===========================")


for generator in sorted(
    test_df[
        test_df["label"] == 1
    ]["generator"]
    .dropna()
    .unique()
):

    mask = (
        (
            test_df["label"] == 1
        )
        &
        (
            test_df["generator"]
            == generator
        )
    )


    if mask.sum() == 0:

        continue


    generator_recall = recall_score(
        y_test[mask],
        y_pred[mask],
        zero_division=0
    )


    generator_accuracy = accuracy_score(
        y_test[mask],
        y_pred[mask]
    )


    print(
        f"{generator:15s} "
        f"Accuracy={generator_accuracy:.4f} "
        f"AI Recall={generator_recall:.4f} "
        f"Samples={mask.sum()}"
    )


# =========================================================
# RESULTS BY LENGTH
# =========================================================

print()
print("PERFORMANCE BY CODE LENGTH")
print("===========================")


for length_group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    mask = (
        test_df[
            "length_group"
        ].values
        == length_group
    )


    if mask.sum() == 0:

        continue


    length_accuracy = accuracy_score(
        y_test[mask],
        y_pred[mask]
    )


    length_precision = precision_score(
        y_test[mask],
        y_pred[mask],
        zero_division=0
    )


    length_recall = recall_score(
        y_test[mask],
        y_pred[mask],
        zero_division=0
    )


    length_f1 = f1_score(
        y_test[mask],
        y_pred[mask],
        zero_division=0
    )


    print(
        f"{length_group:12s} "
        f"Accuracy={length_accuracy:.4f} "
        f"Precision={length_precision:.4f} "
        f"Recall={length_recall:.4f} "
        f"F1={length_f1:.4f} "
        f"Samples={mask.sum()}"
    )


# =========================================================
# SAVE DETAILED TEST RESULTS
# =========================================================

print()
print("SAVING TEST RESULTS")
print("===================")


results_df = test_df.copy()


results_df[
    "predicted_label"
] = y_pred


results_df[
    "predicted_label_name"
] = np.where(
    y_pred == 1,
    "AI",
    "Human"
)


results_df[
    "ai_probability"
] = y_prob


results_df[
    "human_probability"
] = 1.0 - y_prob


results_df[
    "correct"
] = (
    results_df[
        "label"
    ].values
    ==
    y_pred
)


results_path = os.path.join(
    MODEL_DIR,
    "code_test_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# =========================================================
# SAVE MODELS
# =========================================================

print()
print("SAVING CODE MODEL")
print("=================")


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
        "structural_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        structural_scaler,
        f
    )


with open(
    os.path.join(
        MODEL_DIR,
        "code_classifier.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        classifier,
        f
    )


# =========================================================
# SAVE CONFIGURATION
# =========================================================

config = {

    "transformer":
        TRANSFORMER_NAME,

    "word_max_features":
        WORD_MAX_FEATURES,

    "char_max_features":
        CHAR_MAX_FEATURES,

    "word_ngram_range":
        (1, 3),

    "char_ngram_range":
        (2, 6),

    "feature_count":
        X_train.shape[1],

    "train_samples":
        len(train_df),

    "test_samples":
        len(test_df),

    "random_state":
        SEED,

    "specialist":
        "code",

}


with open(
    os.path.join(
        MODEL_DIR,
        "code_config.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        config,
        f
    )


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

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

}


with open(
    os.path.join(
        MODEL_DIR,
        "code_metrics.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        metrics,
        f
    )


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("CODE MODEL SAVED")
print("================")

print(
    f"Directory: {MODEL_DIR}"
)


print()
print("Files:")

print(
    " - word_tfidf_vectorizer.pkl"
)

print(
    " - char_tfidf_vectorizer.pkl"
)

print(
    " - structural_scaler.pkl"
)

print(
    " - code_classifier.pkl"
)

print(
    " - code_config.pkl"
)

print(
    " - code_metrics.pkl"
)

print(
    " - code_test_results.csv"
)


print()
print("CODE-AI SPECIALIST TRAINING COMPLETED")
print("======================================")