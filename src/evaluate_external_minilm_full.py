import os
import re
import joblib
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

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

MODEL_DIR = r"models\text_minilm_full"

TEST_FILE = (
    r"datasets\text\HC3\external_test.csv"
)

BATCH_SIZE = 16
MAX_LENGTH = 256


# ============================================================
# HEADER
# ============================================================

print()
print("EXTERNAL FULL MINILM TEST")
print("=========================")


# ============================================================
# LOAD MODEL
# ============================================================

classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)

linguistic_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "linguistic_scaler.pkl"
    )
)

config = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


MODEL_NAME = config[
    "transformer_name"
]


print(
    "Transformer:",
    MODEL_NAME
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


# ============================================================
# LOAD TRANSFORMER
# ============================================================

print()
print("LOADING MINILM")
print("==============")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

encoder = AutoModel.from_pretrained(
    MODEL_NAME
)

encoder.to(device)
encoder.eval()


print(
    "MiniLM loaded successfully."
)


# ============================================================
# LOAD EXTERNAL DATA
# ============================================================

df = pd.read_csv(
    TEST_FILE
)


df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)

df["label"] = (
    df["label"]
    .astype(int)
)


texts = df["text"].tolist()

y_true = df["label"].values


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
# LINGUISTIC FEATURES
# ============================================================

def linguistic_features(text):

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


    return [
        word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity
    ]


# ============================================================
# CREATE LINGUISTIC MATRIX
# ============================================================

linguistic = np.array(
    [
        linguistic_features(text)
        for text in texts
    ],
    dtype=np.float32
)


linguistic = (
    linguistic_scaler.transform(
        linguistic
    )
)


print()
print(
    "Linguistic feature shape:",
    linguistic.shape
)


# ============================================================
# CREATE MINILM EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    all_embeddings = []

    total = len(texts)


    print()
    print("CREATING MINILM EMBEDDINGS")
    print("==========================")


    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch_texts = texts[
            start:
            start + BATCH_SIZE
        ]


        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )


        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }


        with torch.no_grad():

            outputs = encoder(
                **encoded
            )


        token_embeddings = (
            outputs.last_hidden_state
        )


        attention_mask = (
            encoded["attention_mask"]
        )


        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                token_embeddings.size()
            )
            .float()
        )


        summed = torch.sum(
            token_embeddings * mask,
            dim=1
        )


        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )


        embeddings = (
            summed / counts
        )


        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )


        all_embeddings.append(
            embeddings
            .cpu()
            .numpy()
        )


        processed = min(
            start + BATCH_SIZE,
            total
        )


        print(
            f"Processed "
            f"{processed}/{total}"
        )


    return np.vstack(
        all_embeddings
    ).astype(
        np.float32
    )


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

embeddings = create_embeddings(
    texts
)


print()
print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# COMBINE
# ============================================================

X_external = np.hstack(
    [
        embeddings,
        linguistic
    ]
)


print()
print(
    "Combined feature shape:",
    X_external.shape
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


predictions = classifier.predict(
    X_external
)


probabilities = (
    classifier.predict_proba(
        X_external
    )[:, 1]
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
# RESULTS
# ============================================================

print()
print("EXTERNAL FULL MINILM RESULTS")
print("============================")


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


print(
    confusion_matrix(
        y_true,
        predictions
    )
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
# SAMPLE ANALYSIS
# ============================================================

print()
print("SAMPLE-BY-SAMPLE ANALYSIS")
print("==========================")


for index in range(
    len(df)
):

    actual = (
        "AI"
        if y_true[index] == 1
        else "Human"
    )


    predicted = (
        "AI"
        if predictions[index] == 1
        else "Human"
    )


    ai_probability = float(
        probabilities[index]
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
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "actual_label":
            y_true,

        "predicted_label":
            predictions,

        "ai_probability":
            probabilities,

        "human_probability":
            1.0 - probabilities,

        "correct":
            predictions == y_true,

        "text":
            texts
    }
)


results["actual_name"] = (
    results["actual_label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


results["predicted_name"] = (
    results["predicted_label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


OUTPUT_FILE = (
    r"datasets\text\HC3"
    r"\external_minilm_full_results.csv"
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    "Detailed results saved:"
)

print(
    OUTPUT_FILE
)


print()
print(
    "EXTERNAL FULL MINILM TEST COMPLETED"
)