import math
import os
import re
import pickle

import numpy as np
import torch

from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = r"models\text_final"

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LANGUAGE_MODEL_NAME = "distilgpt2"


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_DIR}"
    )


# ============================================================
# REQUIRED MODEL FILES
# ============================================================

required_files = [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "linguistic_scaler.pkl",
    "text_classifier.pkl",
    "text_config.pkl",
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


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("FINAL AI TEXT DETECTOR")
print("=" * 60)

print()
print("Model: HC3 + RAID + SentenceAI")
print("Model directory:", MODEL_DIR)


# ============================================================
# LOAD WORD TF-IDF
# ============================================================

print()
print("Loading word TF-IDF...")

with open(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    word_vectorizer = pickle.load(f)

print(
    "Word vocabulary:",
    len(word_vectorizer.vocabulary_)
)


# ============================================================
# LOAD CHARACTER TF-IDF
# ============================================================

print("Loading character TF-IDF...")

with open(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    ),
    "rb"
) as f:

    char_vectorizer = pickle.load(f)

print(
    "Character vocabulary:",
    len(char_vectorizer.vocabulary_)
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("Loading final fusion classifier...")

with open(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    ),
    "rb"
) as f:

    classifier = pickle.load(f)


# ============================================================
# LOAD LINGUISTIC SCALER
# ============================================================

print("Loading linguistic scaler...")

with open(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    ),
    "rb"
) as f:

    scaler = pickle.load(f)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    ),
    "rb"
) as f:

    config = pickle.load(f)


print()
print("Model configuration loaded.")


# ============================================================
# LOAD MINILM
# ============================================================

print()
print(
    "Loading transformer:",
    MINILM_NAME
)

print(
    "This may take a moment..."
)

minilm = SentenceTransformer(
    MINILM_NAME
)

print(
    "MiniLM loaded successfully."
)


# ============================================================
# LOAD LANGUAGE MODEL
# ============================================================

print()
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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

language_model.to(device)
language_model.eval()

print(
    "Research-feature device:",
    device
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print()
print("MODEL FEATURE VALIDATION")
print("========================")

word_dimension = len(
    word_vectorizer.vocabulary_
)

char_dimension = len(
    char_vectorizer.vocabulary_
)

embedding_dimension = 384

linguistic_dimension = getattr(
    scaler,
    "n_features_in_",
    5
)

expected_dimension = (
    word_dimension
    + char_dimension
    + embedding_dimension
    + linguistic_dimension
)

classifier_dimension = getattr(
    classifier,
    "n_features_in_",
    None
)

print(
    "Word features       :",
    word_dimension
)

print(
    "Character features  :",
    char_dimension
)

print(
    "MiniLM features     :",
    embedding_dimension
)

print(
    "Linguistic features :",
    linguistic_dimension
)

print(
    "Expected total      :",
    expected_dimension
)

if classifier_dimension is not None:

    print(
        "Classifier expects  :",
        classifier_dimension
    )


if (
    classifier_dimension is not None
    and
    expected_dimension != classifier_dimension
):

    raise RuntimeError(
        "\nFEATURE DIMENSION MISMATCH!\n"
        f"Pipeline creates: {expected_dimension}\n"
        f"Classifier expects: {classifier_dimension}"
    )


print(
    "Feature configuration verified."
)


# ============================================================
# WORD EXTRACTION
# ============================================================

def get_words(text):

    return re.findall(
        r"\b\w+\b",
        str(text)
    )


# ============================================================
# EXACT TRAINING LINGUISTIC FEATURES
# ============================================================

def build_linguistic_features(text):

    words = get_words(text)

    word_count = len(words)


    # --------------------------------------------------------
    # Sentence count
    # --------------------------------------------------------

    sentence_count = max(
        len(
            re.findall(
                r"[.!?]+",
                str(text)
            )
        ),
        1
    )


    # --------------------------------------------------------
    # Average sentence length
    # --------------------------------------------------------

    avg_sentence_length = (
        word_count /
        sentence_count
    )


    # --------------------------------------------------------
    # Vocabulary diversity
    # --------------------------------------------------------

    if word_count > 0:

        vocabulary_diversity = (
            len(
                set(
                    word.lower()
                    for word in words
                )
            )
            /
            word_count
        )

    else:

        vocabulary_diversity = 0.0


    # --------------------------------------------------------
    # EXACT TRAINING LOG WORD COUNT
    # --------------------------------------------------------

    log_word_count = np.log1p(
        word_count
    )


    # --------------------------------------------------------
    # EXACT TRAINING LENGTH BUCKET
    #
    # 0 = very short
    # 1 = short
    # 2 = medium
    # 3 = long
    # --------------------------------------------------------

    if word_count <= 10:

        length_bucket = 0.0

    elif word_count <= 30:

        length_bucket = 1.0

    elif word_count <= 60:

        length_bucket = 2.0

    else:

        length_bucket = 3.0


    # --------------------------------------------------------
    # SAME ORDER AS TRAINING
    # --------------------------------------------------------

    features = np.array(
        [[
            log_word_count,
            sentence_count,
            avg_sentence_length,
            vocabulary_diversity,
            length_bucket
        ]],
        dtype=float
    )

    return features


# ============================================================
# LENGTH GROUP
# ============================================================

def get_length_group(word_count):

    if word_count <= 10:

        return "very_short"

    elif word_count <= 30:

        return "short"

    elif word_count <= 60:

        return "medium"

    else:

        return "long"


# ============================================================
# EVIDENCE LEVEL
# ============================================================

def get_evidence_level(word_count):

    if word_count <= 10:

        return "Very Low"

    elif word_count <= 30:

        return "Low"

    else:

        return "Good"


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_perplexity(text):

    try:

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

        if inputs["input_ids"].shape[1] < 2:

            return None

        with torch.no_grad():

            outputs = language_model(
                **inputs,
                labels=inputs["input_ids"]
            )

        try:

            value = math.exp(
                outputs.loss.item()
            )

        except OverflowError:

            return None

        if math.isfinite(value):

            return value

    except Exception:

        return None

    return None


# ============================================================
# BURSTINESS
# ============================================================

def calculate_burstiness(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text).strip()
    )

    perplexities = []


    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        try:

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

            if inputs["input_ids"].shape[1] < 2:
                continue

            with torch.no_grad():

                outputs = language_model(
                    **inputs,
                    labels=inputs["input_ids"]
                )

            try:

                ppl = math.exp(
                    outputs.loss.item()
                )

            except OverflowError:

                continue

            if math.isfinite(ppl):

                perplexities.append(ppl)

        except Exception:

            continue


    if len(perplexities) < 2:

        return None


    mean_value = (
        sum(perplexities) /
        len(perplexities)
    )

    variance = (
        sum(
            (value - mean_value) ** 2
            for value in perplexities
        )
        /
        len(perplexities)
    )

    result = math.sqrt(
        variance
    )

    if math.isfinite(result):

        return result

    return None


# ============================================================
# RESEARCH FEATURES
# ============================================================

def calculate_research_features(text):

    linguistic = (
        build_linguistic_features(
            text
        )
    )

    word_count = len(
        get_words(text)
    )

    sentence_count = int(
        linguistic[0][1]
    )

    avg_sentence_length = float(
        linguistic[0][2]
    )

    vocabulary_diversity = float(
        linguistic[0][3]
    )

    perplexity = calculate_perplexity(
        text
    )

    burstiness = calculate_burstiness(
        text
    )

    return {

        "word_count":
            word_count,

        "sentence_count":
            sentence_count,

        "avg_sentence_length":
            avg_sentence_length,

        "vocabulary_diversity":
            vocabulary_diversity,

        "perplexity":
            perplexity,

        "burstiness":
            burstiness
    }


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def prepare_features(text):

    # Word TF-IDF
    word_features = (
        word_vectorizer.transform(
            [text]
        )
    )


    # Character TF-IDF
    char_features = (
        char_vectorizer.transform(
            [text]
        )
    )


    # MiniLM
    embedding = minilm.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.ndim == 1:

        embedding = embedding.reshape(
            1,
            -1
        )


    # Exact linguistic features
    linguistic_features = (
        build_linguistic_features(
            text
        )
    )


    # Scale linguistic features
    linguistic_scaled = (
        scaler.transform(
            linguistic_features
        )
    )


    # Combine
    combined = hstack(
        [
            word_features,
            char_features,
            embedding,
            linguistic_scaled
        ],
        format="csr"
    )


    return combined


# ============================================================
# PREDICTION
# ============================================================

def predict_text(text):

    X = prepare_features(
        text
    )


    prediction = classifier.predict(
        X
    )[0]


    probabilities = (
        classifier.predict_proba(
            X
        )[0]
    )


    human_probability = float(
        probabilities[0]
    )

    ai_probability = float(
        probabilities[1]
    )


    confidence = float(
        probabilities[
            int(prediction)
        ]
    )


    if int(prediction) == 1:

        label = "AI-Generated"

    else:

        label = "Human-Written"


    research = (
        calculate_research_features(
            text
        )
    )


    word_count = research[
        "word_count"
    ]


    return {

        "label":
            label,

        "confidence":
            confidence,

        "ai_probability":
            ai_probability,

        "human_probability":
            human_probability,

        "length_group":
            get_length_group(
                word_count
            ),

        "evidence_level":
            get_evidence_level(
                word_count
            ),

        "research":
            research
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(result):

    print()
    print("=" * 60)

    print(
        "TEXT DETECTION RESULT"
    )

    print("=" * 60)

    print()

    print(
        "Prediction:",
        result["label"]
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

    print()

    print(
        "Text Information"
    )

    print(
        "================"
    )

    print(
        "Length group:",
        result["length_group"]
    )

    print(
        "Evidence level:",
        result["evidence_level"]
    )

    # Only show warning for very short text
    if result["length_group"] == "very_short":

        print()
        print(
            "Note: The text is very short."
        )

        print(
            "Short texts contain less stylistic"
        )

        print(
            "evidence, so confidence should be"
        )

        print(
            "interpreted carefully."
        )

    print()

    print(
        "Research Features"
    )

    print(
        "================="
    )

    research = result[
        "research"
    ]

    print(
        "word_count:",
        research["word_count"]
    )

    print(
        "sentence_count:",
        research["sentence_count"]
    )

    print(
        "avg_sentence_length:",
        f"{research['avg_sentence_length']:.4f}"
    )


# ============================================================
# READY
# ============================================================

print()
print("=" * 60)
print("FINAL TEXT DETECTOR READY")
print("=" * 60)

print()
print(
    "Final model: HC3 + RAID + SentenceAI"
)

print(
    "Supports very short, short, medium and long text."
)

print(
    "Features: Word TF-IDF + Character TF-IDF + MiniLM"
)

print()
print(
    "Type 'exit' to stop."
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    print()
    print("-" * 60)

    text = input(
        "Enter text: "
    )


    if text.strip().lower() == "exit":

        print()
        print(
            "Text detector stopped."
        )

        break


    if not text.strip():

        print(
            "Please enter some text."
        )

        continue


    print()
    print(
        "Calculating features..."
    )


    try:

        result = predict_text(
            text
        )

        display_result(
            result
        )

    except Exception as e:

        print()
        print(
            "ERROR:"
        )

        print(
            str(e)
        )