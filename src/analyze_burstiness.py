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

    if inputs["input_ids"].shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

    return math.exp(outputs.loss.item())


def calculate_burstiness(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text).strip()
    )

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

    return burstiness


# Load test data
df = pd.read_csv(TEST_FILE)

# Same 200-sample design as perplexity experiment
human = df[df["label"] == 0].head(100).copy()
ai = df[df["label"] == 1].head(100).copy()

sample = pd.concat([human, ai], ignore_index=True)


print("\nCalculating burstiness for 200 samples...")
print("100 Human + 100 AI")


results = []

for i, row in sample.iterrows():

    burstiness = calculate_burstiness(row["text"])

    if burstiness is not None and math.isfinite(burstiness):
        results.append({
            "label": row["label"],
            "burstiness": burstiness
        })

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/200")


results_df = pd.DataFrame(results)


human_b = results_df[
    results_df["label"] == 0
]["burstiness"]

ai_b = results_df[
    results_df["label"] == 1
]["burstiness"]


print("\nBURSTINESS ANALYSIS")
print("===================")

print("\nHUMAN")
print("Count :", len(human_b))
print("Mean  :", round(human_b.mean(), 2))
print("Median:", round(human_b.median(), 2))
print("Min   :", round(human_b.min(), 2))
print("Max   :", round(human_b.max(), 2))


print("\nAI")
print("Count :", len(ai_b))
print("Mean  :", round(ai_b.mean(), 2))
print("Median:", round(ai_b.median(), 2))
print("Min   :", round(ai_b.min(), 2))
print("Max   :", round(ai_b.max(), 2))