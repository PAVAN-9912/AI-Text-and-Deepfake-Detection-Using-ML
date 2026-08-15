import pandas as pd

INPUT_FILE = r"datasets\text\HC3\validation.csv"
OUTPUT_FILE = r"datasets\text\HC3\feature_validation.csv"


df = pd.read_csv(INPUT_FILE)


# Select 500 Human samples
human = (
    df[df["label"] == 0]
    .sample(n=500, random_state=42)
)


# Select 500 AI samples
ai = (
    df[df["label"] == 1]
    .sample(n=500, random_state=42)
)


# Combine
subset = pd.concat(
    [human, ai],
    ignore_index=True
)


# Shuffle
subset = subset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Create unique sample ID
subset.insert(
    0,
    "sample_id",
    range(len(subset))
)


# Save
subset.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Validation feature subset created.")

print("Total samples:", len(subset))

print("\nLabel distribution:")
print(
    subset["label"]
    .value_counts()
    .sort_index()
)

print("\nSources:")
print(
    subset["source"]
    .value_counts()
)

print("\nOutput:")
print(OUTPUT_FILE)