import math
import os
import re

import joblib
import numpy as np
import pandas as pd
import torch

from scipy.sparse import hstack, csr_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = r"models\text"

TEST_FILE = (
    r"datasets\text\HC3\external_test.csv"
)

LANGUAGE_MODEL_NAME = "distilgpt2"


# ============================================================
# HEADER
# ============================================================

print()
print("EXTERNAL TEXT GENERALIZATION TEST")
print("=================================")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    "tfidf_vectorizer.pkl",
    "research_imputer.pkl",
    "research_scaler.pkl",
    "text_classifier.pkl",
    "text_config.pkl"
]


for filename in required_files:

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing model file: {path}"
        )


if not os.path.exists(TEST_FILE):

    raise FileNotFoundError(
        f"External test file not found: {TEST_FILE}"
    )


# ============================================================
# LOAD SAVED MODEL ARTIFACTS
# ============================================================

print()
print("Loading saved text model...")


vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)


imputer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "research_imputer.pkl"
    )
)


scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "research_scaler.pkl"
    )
)


classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


config = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


research_columns = config[
    "research_columns"
]


# ============================================================
# LOAD DISTILGPT2
# ============================================================

print(
    "Loading language model:",
    LANGUAGE_MODEL_NAME
)


tokenizer = AutoTokenizer.from_pretrained(
    LANGUAGE_MODEL_NAME
)


language_model = (
    AutoModelForCausalLM.from_pretrained(
        LANGUAGE_MODEL_NAME
    )
)


language_model.eval()


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


language_model.to(device)


print(
    "Device:",
    device
)


# ============================================================
# LOAD EXTERNAL DATASET
# ============================================================

df = pd.read_csv(
    TEST_FILE
)


required_columns = [
    "label",
    "text"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


df["label"] = (
    pd.to_numeric(
        df["label"],
        errors="raise"
    )
    .astype(int)
)


print()
print(
    "External samples:",
    len(df)
)


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# BASIC FEATURES
# EXACTLY MATCHES TEXT DETECTOR
# ============================================================

def basic_features(text):

    text = str(text)

    words = re.findall(
        r"\b\w+\b",
        text
    )

    word_count = len(words)


    sentence_count = max(
        len(
            re.findall(
                r"[.!?]+",
                text
            )
        ),
        1
    )


    avg_sentence_length = (
        word_count /
        sentence_count
    )


    unique_words = len(
        set(
            word.lower()
            for word in words
        )
    )


    vocabulary_diversity = (
        unique_words /
        max(
            word_count,
            1
        )
    )


    return {

        "word_count":
            word_count,

        "sentence_count":
            sentence_count,

        "avg_sentence_length":
            avg_sentence_length,

        "vocabulary_diversity":
            vocabulary_diversity
    }


# ============================================================
# PERPLEXITY
# EXACTLY MATCHES TEXT DETECTOR
# ============================================================

def calculate_perplexity(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    if inputs[
        "input_ids"
    ].shape[1] < 2:

        return None


    with torch.no_grad():

        outputs = language_model(
            **inputs,
            labels=inputs[
                "input_ids"
            ]
        )


    value = math.exp(
        outputs.loss.item()
    )


    if math.isfinite(value):

        return value


    return None


# ============================================================
# BURSTINESS
# EXACTLY MATCHES TEXT DETECTOR
# ============================================================

def calculate_burstiness(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text).strip()
    )


    sentence_perplexities = []


    for sentence in sentences:

        sentence = sentence.strip()


        if not sentence:

            continue


        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )


        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        if inputs[
            "input_ids"
        ].shape[1] < 2:

            continue


        with torch.no_grad():

            outputs = language_model(
                **inputs,
                labels=inputs[
                    "input_ids"
                ]
            )


        ppl = math.exp(
            outputs.loss.item()
        )


        if math.isfinite(ppl):

            sentence_perplexities.append(
                ppl
            )


    if len(
        sentence_perplexities
    ) < 2:

        return None


    mean_ppl = (
        sum(sentence_perplexities)
        /
        len(sentence_perplexities)
    )


    variance = (
        sum(
            (
                p - mean_ppl
            ) ** 2
            for p in sentence_perplexities
        )
        /
        len(sentence_perplexities)
    )


    burstiness = math.sqrt(
        variance
    )


    if math.isfinite(
        burstiness
    ):

        return burstiness


    return None


# ============================================================
# EXTRACT ALL RESEARCH FEATURES
# ============================================================

def extract_research_features(text):

    basic = basic_features(
        text
    )


    perplexity = (
        calculate_perplexity(
            text
        )
    )


    burstiness = (
        calculate_burstiness(
            text
        )
    )


    return {

        "word_count":
            basic[
                "word_count"
            ],

        "sentence_count":
            basic[
                "sentence_count"
            ],

        "avg_sentence_length":
            basic[
                "avg_sentence_length"
            ],

        "vocabulary_diversity":
            basic[
                "vocabulary_diversity"
            ],

        "perplexity":
            perplexity,

        "burstiness":
            burstiness
    }


# ============================================================
# PREPARE RESEARCH FEATURES
# EXACTLY MATCHES TEXT DETECTOR
# ============================================================

def prepare_research_features(
    research
):

    research_df = pd.DataFrame(
        [research]
    )


    research_df = research_df[
        research_columns
    ].copy()


    # None -> NaN

    research_df = research_df.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Missing burstiness indicator
    # --------------------------------------------------------

    missing_burstiness = (
        research_df[
            "burstiness"
        ]
        .isna()
        .astype(float)
        .values
        .reshape(-1, 1)
    )


    # --------------------------------------------------------
    # Log transformations
    # --------------------------------------------------------

    research_df[
        "perplexity"
    ] = np.log1p(
        research_df[
            "perplexity"
        ]
    )


    research_df[
        "burstiness"
    ] = np.log1p(
        research_df[
            "burstiness"
        ]
    )


    # --------------------------------------------------------
    # Training-derived imputation
    # --------------------------------------------------------

    research_processed = (
        imputer.transform(
            research_df
        )
    )


    # --------------------------------------------------------
    # Training-derived standardization
    # --------------------------------------------------------

    research_processed = (
        scaler.transform(
            research_processed
        )
    )


    # --------------------------------------------------------
    # Add missing indicator
    # --------------------------------------------------------

    research_processed = np.hstack(
        [
            research_processed,
            missing_burstiness
        ]
    )


    return research_processed


# ============================================================
# PROCESS EXTERNAL DATA
# ============================================================

print()
print("CALCULATING FEATURES")
print("====================")


tfidf_matrix = vectorizer.transform(
    df["text"].fillna("")
)


research_vectors = []

research_records = []


total = len(df)


for index, text in enumerate(
    df["text"].fillna("")
):

    text = str(text)


    research = (
        extract_research_features(
            text
        )
    )


    research_processed = (
        prepare_research_features(
            research
        )
    )


    research_vectors.append(
        research_processed[0]
    )


    research_records.append(
        research
    )


    current = index + 1


    print(
        f"Processed "
        f"{current}/{total}"
    )


research_matrix = np.asarray(
    research_vectors,
    dtype=np.float64
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_combined = hstack(
    [
        tfidf_matrix,

        csr_matrix(
            research_matrix
        )
    ]
).tocsr()


print()
print(
    "TF-IDF shape:",
    tfidf_matrix.shape
)


print(
    "Research shape:",
    research_matrix.shape
)


print(
    "Combined shape:",
    X_combined.shape
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("RUNNING MODEL")
print("=============")


y_true = (
    df["label"]
    .values
)


predictions = classifier.predict(
    X_combined
)


probabilities = (
    classifier.predict_proba(
        X_combined
    )
)


ai_probabilities = (
    probabilities[:, 1]
)


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print("EXTERNAL TEST RESULTS")
print("=====================")


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


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")


cm = confusion_matrix(
    y_true,
    predictions
)


print(
    cm
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# SAMPLE-BY-SAMPLE RESULTS
# ============================================================

print()
print("SAMPLE-BY-SAMPLE ANALYSIS")
print("==========================")


for index, row in df.iterrows():

    actual = (
        "AI"
        if int(row["label"]) == 1
        else "Human"
    )


    predicted = (
        "AI"
        if int(predictions[index]) == 1
        else "Human"
    )


    ai_probability = (
        float(
            ai_probabilities[index]
        )
    )


    confidence = max(
        ai_probability,
        1.0 - ai_probability
    )


    result = (
        "CORRECT"
        if actual == predicted
        else "WRONG"
    )


    print(
        f"{index + 1:02d}. "
        f"Actual={actual:<6} "
        f"Predicted={predicted:<6} "
        f"AI={ai_probability * 100:6.2f}% "
        f"Confidence={confidence * 100:6.2f}% "
        f"{result}"
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(
    {
        "actual_label":
            y_true,

        "predicted_label":
            predictions,

        "ai_probability":
            ai_probabilities,

        "human_probability":
            1.0 - ai_probabilities,

        "correct":
            predictions == y_true
    }
)


results_df[
    "actual_name"
] = results_df[
    "actual_label"
].map(
    {
        0: "Human",
        1: "AI"
    }
)


results_df[
    "predicted_name"
] = results_df[
    "predicted_label"
].map(
    {
        0: "Human",
        1: "AI"
    }
)


results_df[
    "text"
] = df[
    "text"
].values


RESULT_FILE = (
    r"datasets\text\HC3"
    r"\external_test_results.csv"
)


results_df.to_csv(
    RESULT_FILE,
    index=False
)


print()
print(
    "Detailed results saved:"
)

print(
    RESULT_FILE
)


print()
print(
    "EXTERNAL GENERALIZATION TEST COMPLETED"
)