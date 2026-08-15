import math
import os
import re

import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "distilgpt2"

INPUT_FILE = r"datasets\text\HC3\feature_development.csv"
OUTPUT_FILE = r"datasets\text\HC3\research_features.csv"

CHECKPOINT_EVERY = 50


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

    value = math.exp(outputs.loss.item())

    if math.isfinite(value):
        return value

    return None


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

    if math.isfinite(burstiness):
        return burstiness

    return None


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Create guaranteed unique sample IDs
df.insert(0, "sample_id", range(len(df)))

print("Total samples:", len(df))


# --------------------------------------------------
# Resume support
# --------------------------------------------------

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_csv(OUTPUT_FILE)

    processed_ids = set(
        existing["sample_id"].astype(int)
    )

    print(
        "Existing results found:",
        len(existing)
    )

else:

    existing = pd.DataFrame()

    processed_ids = set()

    print("Starting feature extraction from scratch.")


results = []


# --------------------------------------------------
# Process samples
# --------------------------------------------------

for _, row in df.iterrows():

    sample_id = int(row["sample_id"])

    if sample_id in processed_ids:
        continue

    text = str(row["text"])

    basic = basic_features(text)

    perplexity = calculate_perplexity(text)

    burstiness = calculate_burstiness(text)

    result = {

        "sample_id": sample_id,

        "record_id": row["record_id"],

        "label": row["label"],

        "source": row["source"],

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
    }

    results.append(result)

    processed_ids.add(sample_id)


    # Save checkpoint
    if len(results) >= CHECKPOINT_EVERY:

        checkpoint = pd.DataFrame(results)

        if not existing.empty:

            output = pd.concat(
                [existing, checkpoint],
                ignore_index=True
            )

        else:

            output = checkpoint


        output.to_csv(
            OUTPUT_FILE,
            index=False
        )

        existing = output

        results = []

        print(
            f"Checkpoint saved: "
            f"{len(existing)}/{len(df)}"
        )


# --------------------------------------------------
# Save remaining results
# --------------------------------------------------

if results:

    checkpoint = pd.DataFrame(results)

    if not existing.empty:

        output = pd.concat(
            [existing, checkpoint],
            ignore_index=True
        )

    else:

        output = checkpoint


    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    existing = output


print("\nFEATURE EXTRACTION COMPLETED")

print("Total feature rows:", len(existing))

print("Unique sample IDs:", existing["sample_id"].nunique())

print("Output:", OUTPUT_FILE)

print("\nMissing values:")

print(existing.isna().sum())