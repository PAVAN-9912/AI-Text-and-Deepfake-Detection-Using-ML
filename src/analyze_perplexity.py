import math
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "distilgpt2"
TEST_FILE = r"datasets\text\HC3\test.csv"


print("Loading language model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()


def calculate_perplexity(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

    return math.exp(outputs.loss.item())


# Load test data
df = pd.read_csv(TEST_FILE)

# Take 100 Human + 100 AI
human = df[df["label"] == 0].head(100).copy()
ai = df[df["label"] == 1].head(100).copy()

sample = pd.concat([human, ai], ignore_index=True)


print("\nCalculating perplexity for 200 samples...")
print("100 Human + 100 AI")


results = []

for i, row in sample.iterrows():

    text = str(row["text"])

    perplexity = calculate_perplexity(text)

    results.append({
        "label": row["label"],
        "perplexity": perplexity
    })

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/200")


results_df = pd.DataFrame(results)


human_ppl = results_df[
    results_df["label"] == 0
]["perplexity"]

ai_ppl = results_df[
    results_df["label"] == 1
]["perplexity"]


print("\nPERPLEXITY ANALYSIS")
print("===================")

print("\nHUMAN")
print("Count :", len(human_ppl))
print("Mean  :", round(human_ppl.mean(), 2))
print("Median:", round(human_ppl.median(), 2))
print("Min   :", round(human_ppl.min(), 2))
print("Max   :", round(human_ppl.max(), 2))


print("\nAI")
print("Count :", len(ai_ppl))
print("Mean  :", round(ai_ppl.mean(), 2))
print("Median:", round(ai_ppl.median(), 2))
print("Min   :", round(ai_ppl.min(), 2))
print("Max   :", round(ai_ppl.max(), 2))