import pandas as pd
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

TRAIN_PATH = r"datasets\text\HC3\train.csv"
VAL_PATH = r"datasets\text\HC3\validation.csv"


# =========================================================
# FUNCTIONS
# =========================================================

def word_count(text):
    return len(str(text).split())


def add_length_group(df):
    df = df.copy()

    df["word_count"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .apply(word_count)
    )

    def length_group(n):
        if n <= 30:
            return "short"
        elif n <= 100:
            return "medium"
        else:
            return "long"

    df["length_group"] = (
        df["word_count"]
        .apply(length_group)
    )

    return df


# =========================================================
# LOAD DATA
# =========================================================

print()
print("HC3 TEXT LENGTH ANALYSIS")
print("=========================")

print()
print("Loading datasets...")


train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)


print(
    f"Training samples  : {len(train)}"
)

print(
    f"Validation samples: {len(validation)}"
)


# =========================================================
# ADD LENGTH GROUPS
# =========================================================

train = add_length_group(train)
validation = add_length_group(validation)


# =========================================================
# TRAINING DISTRIBUTION
# =========================================================

print()
print("TRAINING LENGTH DISTRIBUTION")
print("============================")


train_table = pd.crosstab(
    train["length_group"],
    train["label"]
)

train_table = train_table.rename(
    columns={
        0: "Human",
        1: "AI"
    }
)

print(
    train_table.to_string()
)


# =========================================================
# TRAINING PERCENTAGES
# =========================================================

print()
print("TRAINING LENGTH PERCENTAGES")
print("===========================")


for group in [
    "short",
    "medium",
    "long"
]:

    subset = train[
        train["length_group"] == group
    ]

    if len(subset) == 0:
        continue

    human = (
        subset["label"] == 0
    ).sum()

    ai = (
        subset["label"] == 1
    ).sum()

    print()
    print(
        f"{group.upper()}"
    )

    print(
        f"Total : {len(subset)}"
    )

    print(
        f"Human : {human} "
        f"({human / len(subset) * 100:.2f}%)"
    )

    print(
        f"AI    : {ai} "
        f"({ai / len(subset) * 100:.2f}%)"
    )


# =========================================================
# VALIDATION DISTRIBUTION
# =========================================================

print()
print("VALIDATION LENGTH DISTRIBUTION")
print("==============================")


val_table = pd.crosstab(
    validation["length_group"],
    validation["label"]
)

val_table = val_table.rename(
    columns={
        0: "Human",
        1: "AI"
    }
)

print(
    val_table.to_string()
)


# =========================================================
# DETAILED STATISTICS
# =========================================================

print()
print("DETAILED TRAINING STATISTICS")
print("============================")


for label, name in [
    (0, "Human"),
    (1, "AI")
]:

    subset = train[
        train["label"] == label
    ]


    print()
    print(name)
    print(
        "Samples:",
        len(subset)
    )

    print(
        "Average words:",
        f"{subset['word_count'].mean():.2f}"
    )

    print(
        "Median words:",
        f"{subset['word_count'].median():.2f}"
    )

    print(
        "Minimum words:",
        subset["word_count"].min()
    )

    print(
        "Maximum words:",
        subset["word_count"].max()
    )


# =========================================================
# SOURCE × LENGTH × LABEL
# =========================================================

print()
print("SOURCE × LENGTH × LABEL")
print("========================")


source_table = pd.crosstab(
    [
        train["source"],
        train["length_group"]
    ],
    train["label"]
)

source_table = source_table.rename(
    columns={
        0: "Human",
        1: "AI"
    }
)

print(
    source_table.to_string()
)


# =========================================================
# SAVE ANALYSIS
# =========================================================

output_path = (
    r"datasets\text\analysis"
    r"\text_length_analysis.csv"
)


summary_rows = []


for dataset_name, data in [
    ("train", train),
    ("validation", validation)
]:

    for group in [
        "short",
        "medium",
        "long"
    ]:

        for label, label_name in [
            (0, "Human"),
            (1, "AI")
        ]:

            subset = data[
                (data["length_group"] == group)
                &
                (data["label"] == label)
            ]

            summary_rows.append(
                {
                    "dataset":
                        dataset_name,

                    "length_group":
                        group,

                    "label":
                        label_name,

                    "samples":
                        len(subset),

                    "average_words":
                        (
                            subset["word_count"].mean()
                            if len(subset)
                            else 0
                        ),

                    "median_words":
                        (
                            subset["word_count"].median()
                            if len(subset)
                            else 0
                        )
                }
            )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    output_path,
    index=False
)


print()
print("ANALYSIS SAVED")
print("==============")

print(
    f"Output: {output_path}"
)


print()
print(
    "TEXT LENGTH ANALYSIS COMPLETED"
)