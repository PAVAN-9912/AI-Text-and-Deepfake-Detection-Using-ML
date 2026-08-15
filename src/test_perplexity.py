import math
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "distilgpt2"
TEST_FILE = r"datasets\text\HC3\test.csv"


print("Loading model...")

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

    loss = outputs.loss

    return math.exp(loss.item())


# Load test data
df = pd.read_csv(TEST_FILE)

# Select two human and two AI samples
human_samples = df[df["label"] == 0].head(2)
ai_samples = df[df["label"] == 1].head(2)


print("\nPERPLEXITY RESULTS")
print("==================")


for _, row in pd.concat([human_samples, ai_samples]).iterrows():

    text = row["text"]

    perplexity = calculate_perplexity(text)

    label = "HUMAN" if row["label"] == 0 else "AI"

    print("\nLabel:", label)
    print("Perplexity:", round(perplexity, 2))
    print("Text:", text[:200].replace("\n", " "))