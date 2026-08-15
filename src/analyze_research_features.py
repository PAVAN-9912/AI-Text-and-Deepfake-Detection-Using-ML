import pandas as pd

FILE = r"datasets\text\HC3\research_features.csv"

df = pd.read_csv(FILE)

print("RESEARCH FEATURE DATASET")
print("========================")

print("Rows:", len(df))
print("Unique samples:", df["sample_id"].nunique())

print("\nLabels:")
print(df["label"].value_counts().sort_index())

print("\nMissing values:")
print(df.isna().sum())


# --------------------------------------------------
# Feature averages by class
# --------------------------------------------------

features = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity",
    "perplexity",
    "burstiness"
]

print("\nFEATURE MEANS BY LABEL")
print("=====================")

print(
    df.groupby("label")[features]
    .mean()
    .round(3)
    .to_string()
)


# --------------------------------------------------
# Feature medians
# --------------------------------------------------

print("\nFEATURE MEDIANS BY LABEL")
print("=======================")

print(
    df.groupby("label")[features]
    .median()
    .round(3)
    .to_string()
)


# --------------------------------------------------
# Perplexity ranges
# --------------------------------------------------

print("\nPERPLEXITY")
print("==========")

for label, name in [(0, "HUMAN"), (1, "AI")]:

    values = df[df["label"] == label]["perplexity"].dropna()

    print(
        f"{name}: "
        f"Count={len(values)}, "
        f"Min={values.min():.2f}, "
        f"Median={values.median():.2f}, "
        f"Max={values.max():.2f}"
    )


# --------------------------------------------------
# Burstiness ranges
# --------------------------------------------------

print("\nBURSTINESS")
print("==========")

for label, name in [(0, "HUMAN"), (1, "AI")]:

    values = df[df["label"] == label]["burstiness"].dropna()

    print(
        f"{name}: "
        f"Count={len(values)}, "
        f"Min={values.min():.2f}, "
        f"Median={values.median():.2f}, "
        f"Max={values.max():.2f}"
    )


# --------------------------------------------------
# Missing burstiness by label
# --------------------------------------------------

print("\nMISSING BURSTINESS BY LABEL")
print("==========================")

print(
    df.groupby("label")["burstiness"]
    .apply(lambda x: x.isna().sum())
)


# --------------------------------------------------
# Extreme burstiness
# --------------------------------------------------

print("\nTOP 10 BURSTINESS VALUES")
print("========================")

print(
    df[
        ["sample_id", "label", "burstiness"]
    ]
    .dropna()
    .sort_values(
        "burstiness",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)