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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = r"models\text"

LANGUAGE_MODEL_NAME = "distilgpt2"


# ============================================================
# CHECK MODEL FILES
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
            f"\nMissing model file:\n{path}\n\n"
            "Run:\n"
            "python src\\train_text_final.py"
            "\nfirst."
        )


# ============================================================
# LOAD SAVED TEXT MODEL
# ============================================================

print("AI TEXT DETECTOR")
print("================")

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


language_model = AutoModelForCausalLM.from_pretrained(
    LANGUAGE_MODEL_NAME
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

print(
    "Text model loaded successfully."
)


# ============================================================
# BASIC FEATURES
# EXACTLY MATCHES TRAINING FEATURE EXTRACTION
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
        max(word_count, 1)
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


    # --------------------------------------------------------
    # IMPORTANT:
    # A single sentence cannot produce burstiness.
    # Therefore return None.
    # This will later be converted to NaN.
    # --------------------------------------------------------

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
            (p - mean_ppl) ** 2
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
# CALCULATE ALL RESEARCH FEATURES
# ============================================================

def extract_research_features(text):

    basic = basic_features(
        text
    )


    perplexity = calculate_perplexity(
        text
    )


    burstiness = calculate_burstiness(
        text
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


    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # During inference, short text can produce Python None.
    # During training, Pandas represents missing values as NaN.
    #
    # Convert None -> NaN before log1p().
    # --------------------------------------------------------

    research_df = research_df.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Burstiness missing indicator
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
    # LOG TRANSFORMATIONS
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
    # TRAINING-DERIVED IMPUTATION
    # --------------------------------------------------------

    research_processed = (
        imputer.transform(
            research_df
        )
    )


    # --------------------------------------------------------
    # TRAINING-DERIVED STANDARDIZATION
    # --------------------------------------------------------

    research_processed = (
        scaler.transform(
            research_processed
        )
    )


    # --------------------------------------------------------
    # ADD BURSTINESS MISSING INDICATOR
    # --------------------------------------------------------

    research_processed = np.hstack(
        [
            research_processed,
            missing_burstiness
        ]
    )


    return research_processed


# ============================================================
# PREDICTION
# ============================================================

def predict_text(text):

    # --------------------------------------------------------
    # TF-IDF FEATURES
    # --------------------------------------------------------

    X_tfidf = vectorizer.transform(
        [text]
    )


    # --------------------------------------------------------
    # RESEARCH FEATURES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COMBINE TF-IDF + RESEARCH FEATURES
    # --------------------------------------------------------

    X_combined = hstack(
        [
            X_tfidf,

            csr_matrix(
                research_processed
            )
        ]
    ).tocsr()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = classifier.predict(
        X_combined
    )[0]


    probabilities = (
        classifier.predict_proba(
            X_combined
        )[0]
    )


    human_probability = float(
        probabilities[0]
    )


    ai_probability = float(
        probabilities[1]
    )


    confidence = float(
        probabilities[prediction]
    )


    if prediction == 1:

        label = "AI-Generated"

    else:

        label = "Human-Written"


    return {

        "label":
            label,

        "prediction":
            int(prediction),

        "confidence":
            confidence,

        "ai_probability":
            ai_probability,

        "human_probability":
            human_probability,

        "research_features":
            research
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    text,
    result
):

    print("\n")
    print("=" * 60)

    print(
        "TEXT DETECTION RESULT"
    )

    print("=" * 60)


    print(
        "\nPrediction:",
        result[
            "label"
        ]
    )


    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )


    print(
        "AI probability:",
        f"{result['ai_probability'] * 100:.2f}%"
    )


    print(
        "Human probability:",
        f"{result['human_probability'] * 100:.2f}%"
    )


    print(
        "\nResearch Features"
    )

    print(
        "================="
    )


    for key, value in result[
        "research_features"
    ].items():

        if value is None:

            print(
                f"{key}: None"
            )


        elif isinstance(
            value,
            float
        ):

            print(
                f"{key}: {value:.4f}"
            )


        else:

            print(
                f"{key}: {value}"
            )


    print(
        "=" * 60
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

print(
    "\nTEXT DETECTOR READY"
)

print(
    "==================="
)


print(
    "Enter text to analyze."
)


print(
    "Type 'exit' to stop."
)


while True:

    print(
        "\n" + "-" * 60
    )


    text = input(
        "Enter text: "
    ).strip()


    if text.lower() == "exit":

        print(
            "\nExiting detector."
        )

        break


    if not text:

        print(
            "Please enter some text."
        )

        continue


    print(
        "\nCalculating research features..."
    )


    result = predict_text(
        text
    )


    display_result(
        text,
        result
    )