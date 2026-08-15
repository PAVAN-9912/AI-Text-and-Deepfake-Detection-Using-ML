import pandas as pd

INPUT_FILE = r"datasets\text\HC3\train.csv"
OUTPUT_FILE = r"datasets\text\HC3\feature_development.csv"

df = pd.read_csv(INPUT_FILE)

# Take 1,000 Human samples
human = (
    df[df["label"] == 0]
    .sample(n=1000, random_state=42)
)

# Take 1,000 AI samples
ai = (
    df[df["label"] == 1]
    .sample(n=1000, random_state=42)
)

# Combine and shuffle
subset = pd.concat(
    [human, ai],
    ignore_index=True
)

subset = subset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
subset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Feature-development subset created.")
print("Total samples:", len(subset))

print("\nLabel distribution:")
print(subset["label"].value_counts())

print("\nSources:")
print(subset["source"].value_counts())

print("\nOutput:")
print(OUTPUT_FILE)