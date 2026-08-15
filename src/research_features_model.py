import math
import re

import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


MODEL_NAME = "distilgpt2"
TEST_FILE = r"datasets\text\HC3\test.csv"


print("Loading language model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()


# --------------------------------------------------
# Basic statistical features
# --------------------------------------------------

def basic_features(text):

    text = str(text)

    words = re.findall(r"\b\w+\b", text)

    word_count = len(words)

    sentence_count = max(
        len(re.findall(r"[.!?]+", text)),
        1
    )

    avg_sentence_length = (
        word_count / sentence_count
    )

    unique_words = len(
        set(word.lower() for word in words)
    )

    vocabulary_diversity = (
        unique_words / max(word_count, 1)
    )

    return (
        word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity
    )


# --------------------------------------------------
# Perplexity
# --------------------------------------------------

def calculate_perplexity(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    if inputs["input_ids"].shape[1] < 2:
        return None

    with torch.no_grad():

        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

    value = math.exp(outputs.loss.item())

    return value if math.isfinite(value) else None


# --------------------------------------------------
# Burstiness
# --------------------------------------------------

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

        if inputs["input_ids"].shape[1] < 2:
            continue

        with torch.no_grad():

            outputs = model(
                **inputs,
                labels=inputs["input_ids"]
            )

        ppl = math.exp(outputs.loss.item())

        if math.isfinite(ppl):
            sentence_perplexities.append(ppl)

    if len(sentence_perplexities) < 2:
        return None

    mean_ppl = sum(sentence_perplexities) / len(
        sentence_perplexities
    )

    variance = sum(
        (p - mean_ppl) ** 2
        for p in sentence_perplexities
    ) / len(sentence_perplexities)

    burstiness = math.sqrt(variance)

    return burstiness if math.isfinite(burstiness) else None


# --------------------------------------------------
# Load 200 samples
# --------------------------------------------------

df = pd.read_csv(TEST_FILE)

human = df[df["label"] == 0].head(100)
ai = df[df["label"] == 1].head(100)

sample = pd.concat(
    [human, ai],
    ignore_index=True
)


print("\nExtracting research features...")
print("100 Human + 100 AI")


results = []


for i, row in sample.iterrows():

    text = str(row["text"])

    basic = basic_features(text)

    perplexity = calculate_perplexity(text)

    burstiness = calculate_burstiness(text)

    if perplexity is None or burstiness is None:
        print(
            f"Skipping sample {i + 1}: insufficient features"
        )
        continue

    results.append({

        "word_count": basic[0],

        "sentence_count": basic[1],

        "avg_sentence_length": basic[2],

        "vocabulary_diversity": basic[3],

        # Log transformation reduces extreme outliers
        "log_perplexity": math.log1p(perplexity),

        "log_burstiness": math.log1p(burstiness),

        "label": row["label"]
    })

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/200")


features = pd.DataFrame(results)


print("\nUsable samples:", len(features))

print("\nFeature summary:")
print(
    features.groupby("label").mean().round(3)
)


# --------------------------------------------------
# Prepare X and y
# --------------------------------------------------

X = features.drop(columns=["label"])
y = features["label"]


# --------------------------------------------------
# 5-fold cross-validation
# --------------------------------------------------

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score)
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results_cv = cross_validate(
    model_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring
)


print("\n5-FOLD CROSS-VALIDATION")
print("=======================")

print(
    "Accuracy :",
    round(results_cv["test_accuracy"].mean(), 4)
)

print(
    "Precision:",
    round(results_cv["test_precision"].mean(), 4)
)

print(
    "Recall   :",
    round(results_cv["test_recall"].mean(), 4)
)

print(
    "F1 Score :",
    round(results_cv["test_f1"].mean(), 4)
)