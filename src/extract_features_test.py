import math
import re

import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


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

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_diversity": vocabulary_diversity
    }


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

    return math.exp(outputs.loss.item())


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

    return math.sqrt(variance)


# --------------------------------------------------
# Load only 20 samples
# --------------------------------------------------

df = pd.read_csv(TEST_FILE)

human = df[df["label"] == 0].head(10)
ai = df[df["label"] == 1].head(10)

sample = pd.concat(
    [human, ai],
    ignore_index=True
)


print("\nExtracting features...")
print("10 Human + 10 AI")


results = []


for i, row in sample.iterrows():

    text = str(row["text"])

    basic = basic_features(text)

    perplexity = calculate_perplexity(text)

    burstiness = calculate_burstiness(text)

    results.append({

        "label": row["label"],

        "word_count":
            basic["word_count"],

        "sentence_count":
            basic["sentence_count"],

        "avg_sentence_length":
            basic["avg_sentence_length"],

        "vocabulary_diversity":
            basic["vocabulary_diversity"],

        "perplexity":
            perplexity,

        "burstiness":
            burstiness
    })

    print(
        f"Processed {i + 1}/20"
    )


features = pd.DataFrame(results)


print("\nFEATURE TABLE")
print("=============")

print(
    features.round(3).to_string(
        index=False
    )
)


print("\nMissing values")
print("==============")

print(
    features.isna().sum()
)