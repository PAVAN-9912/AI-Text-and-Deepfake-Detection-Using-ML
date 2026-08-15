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


def sentence_perplexity(sentence):
    sentence = sentence.strip()

    if not sentence:
        return None

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    # Ignore very short token sequences
    if inputs["input_ids"].shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

    return math.exp(outputs.loss.item())


def analyze_text(text):
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())

    perplexities = []

    for sentence in sentences:
        value = sentence_perplexity(sentence)

        if value is not None and math.isfinite(value):
            perplexities.append(value)

    if len(perplexities) < 2:
        return None

    mean_ppl = sum(perplexities) / len(perplexities)

    variance = sum(
        (p - mean_ppl) ** 2
        for p in perplexities
    ) / len(perplexities)

    burstiness = math.sqrt(variance)

    return {
        "sentence_count": len(perplexities),
        "mean_perplexity": mean_ppl,
        "burstiness": burstiness,
        "sentence_perplexities": perplexities
    }


# Load test data
df = pd.read_csv(TEST_FILE)

# Two Human + two AI samples
human = df[df["label"] == 0].head(2)
ai = df[df["label"] == 1].head(2)

samples = pd.concat([human, ai])


print("\nBURSTINESS ANALYSIS")
print("===================")


for _, row in samples.iterrows():

    result = analyze_text(row["text"])

    label = "HUMAN" if row["label"] == 0 else "AI"

    print("\nLabel:", label)

    if result is None:
        print("Not enough sentences for analysis.")
        continue

    print("Sentence count:", result["sentence_count"])
    print("Mean perplexity:", round(result["mean_perplexity"], 2))
    print("Burstiness:", round(result["burstiness"], 2))
    print(
        "Sentence perplexities:",
        [round(x, 2) for x in result["sentence_perplexities"]]
    )