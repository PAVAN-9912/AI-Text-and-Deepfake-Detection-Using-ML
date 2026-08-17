import os
import re
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

HC3_TRAIN = r"datasets\text\HC3\train.csv"
HC3_VALIDATION = r"datasets\text\HC3\validation.csv"
RAID_DATA = r"datasets\text\RAID\raid_external.csv"


# ============================================================
# HEADER
# ============================================================

print()
print("TEXT DATASET SHIFT ANALYSIS")
print("===========================")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("LOADING DATA")
print("============")


train = pd.read_csv(
    HC3_TRAIN
)

validation = pd.read_csv(
    HC3_VALIDATION
)

raid = pd.read_csv(
    RAID_DATA
)


print(
    "HC3 training samples:",
    len(train)
)

print(
    "HC3 validation samples:",
    len(validation)
)

print(
    "RAID samples:",
    len(raid)
)


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_statistics(text):

    text = str(text)

    words = re.findall(
        r"\b\w+\b",
        text
    )

    sentences = re.findall(
        r"[.!?]+",
        text
    )

    characters = len(
        text
    )

    word_count = len(
        words
    )

    sentence_count = max(
        len(sentences),
        1
    )

    unique_words = len(
        set(
            w.lower()
            for w in words
        )
    )

    vocabulary_diversity = (
        unique_words /
        max(
            word_count,
            1
        )
    )

    return {
        "characters": characters,
        "words": word_count,
        "sentences": sentence_count,
        "vocabulary_diversity":
            vocabulary_diversity
    }


# ============================================================
# ADD STATISTICS
# ============================================================

def add_statistics(df):

    stats = df["text"].fillna("").apply(
        text_statistics
    )

    stats_df = pd.DataFrame(
        stats.tolist()
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            stats_df
        ],
        axis=1
    )


print()
print("CALCULATING TEXT STATISTICS")
print("===========================")


train_stats = add_statistics(
    train
)

validation_stats = add_statistics(
    validation
)

raid_stats = add_statistics(
    raid
)


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(
    name,
    df
):

    print()
    print(
        name
    )

    print(
        "-" * len(name)
    )

    print(
        "Samples:",
        len(df)
    )

    print(
        "Average characters:",
        f"{df['characters'].mean():.2f}"
    )

    print(
        "Median characters:",
        f"{df['characters'].median():.2f}"
    )

    print(
        "Average words:",
        f"{df['words'].mean():.2f}"
    )

    print(
        "Median words:",
        f"{df['words'].median():.2f}"
    )

    print(
        "Average sentences:",
        f"{df['sentences'].mean():.2f}"
    )

    print(
        "Average vocabulary diversity:",
        f"{df['vocabulary_diversity'].mean():.4f}"
    )


print_summary(
    "HC3 TRAINING",
    train_stats
)

print_summary(
    "HC3 VALIDATION",
    validation_stats
)

print_summary(
    "RAID EXTERNAL",
    raid_stats
)


# ============================================================
# HUMAN VS AI STATISTICS
# ============================================================

def class_summary(
    name,
    df
):

    print()
    print(
        name
    )

    print(
        "=" * len(name)
    )


    for label in sorted(
        df["label"].unique()
    ):

        subset = df[
            df["label"] == label
        ]


        label_name = (
            "Human"
            if label == 0
            else "AI"
        )


        print()
        print(
            label_name
        )

        print(
            "Samples:",
            len(subset)
        )

        print(
            "Average words:",
            f"{subset['words'].mean():.2f}"
        )

        print(
            "Median words:",
            f"{subset['words'].median():.2f}"
        )

        print(
            "Average characters:",
            f"{subset['characters'].mean():.2f}"
        )

        print(
            "Average sentences:",
            f"{subset['sentences'].mean():.2f}"
        )

        print(
            "Average vocabulary diversity:",
            f"{subset['vocabulary_diversity'].mean():.4f}"
        )


class_summary(
    "HC3 TRAINING — HUMAN VS AI",
    train_stats
)

class_summary(
    "HC3 VALIDATION — HUMAN VS AI",
    validation_stats
)

class_summary(
    "RAID — HUMAN VS AI",
    raid_stats
)


# ============================================================
# HC3 SOURCE DISTRIBUTION
# ============================================================

print()
print("HC3 TRAINING SOURCE DISTRIBUTION")
print("================================")

print(
    train.groupby(
        [
            "source",
            "label"
        ]
    ).size().unstack(
        fill_value=0
    ).to_string()
)


print()
print("HC3 VALIDATION SOURCE DISTRIBUTION")
print("===================================")

print(
    validation.groupby(
        [
            "source",
            "label"
        ]
    ).size().unstack(
        fill_value=0
    ).to_string()
)


# ============================================================
# RAID MODEL DISTRIBUTION
# ============================================================

print()
print("RAID GENERATOR DISTRIBUTION")
print("============================")

print(
    raid.groupby(
        [
            "model",
            "label"
        ]
    ).size().unstack(
        fill_value=0
    ).to_string()
)


# ============================================================
# RAID DOMAIN DISTRIBUTION
# ============================================================

print()
print("RAID DOMAIN DISTRIBUTION")
print("========================")

print(
    raid.groupby(
        [
            "domain",
            "label"
        ]
    ).size().unstack(
        fill_value=0
    ).to_string()
)


# ============================================================
# RAID MODEL STATISTICS
# ============================================================

print()
print("RAID STATISTICS BY GENERATOR")
print("============================")


for model in sorted(
    raid_stats["model"].unique()
):

    subset = raid_stats[
        raid_stats["model"] == model
    ]


    print()
    print(
        f"MODEL: {model}"
    )

    print(
        "Samples:",
        len(subset)
    )

    print(
        "Average words:",
        f"{subset['words'].mean():.2f}"
    )

    print(
        "Median words:",
        f"{subset['words'].median():.2f}"
    )

    print(
        "Average characters:",
        f"{subset['characters'].mean():.2f}"
    )

    print(
        "Average sentences:",
        f"{subset['sentences'].mean():.2f}"
    )

    print(
        "Average vocabulary diversity:",
        f"{subset['vocabulary_diversity'].mean():.4f}"
    )


# ============================================================
# RAID DOMAIN STATISTICS
# ============================================================

print()
print("RAID STATISTICS BY DOMAIN")
print("=========================")


for domain in sorted(
    raid_stats["domain"].unique()
):

    subset = raid_stats[
        raid_stats["domain"] == domain
    ]


    print()
    print(
        f"DOMAIN: {domain}"
    )

    print(
        "Samples:",
        len(subset)
    )

    print(
        "Average words:",
        f"{subset['words'].mean():.2f}"
    )

    print(
        "Median words:",
        f"{subset['words'].median():.2f}"
    )

    print(
        "Average characters:",
        f"{subset['characters'].mean():.2f}"
    )

    print(
        "Average vocabulary diversity:",
        f"{subset['vocabulary_diversity'].mean():.4f}"
    )


# ============================================================
# LABEL BALANCE
# ============================================================

print()
print("LABEL BALANCE")
print("=============")


for name, df in [
    ("HC3 TRAIN", train),
    ("HC3 VALIDATION", validation),
    ("RAID", raid)
]:

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )


    total = len(df)


    human_percentage = (
        counts.get(0, 0) /
        total *
        100
    )


    ai_percentage = (
        counts.get(1, 0) /
        total *
        100
    )


    print()
    print(
        name
    )

    print(
        f"Human: {human_percentage:.2f}%"
    )

    print(
        f"AI   : {ai_percentage:.2f}%"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

output_dir = r"datasets\text\analysis"

os.makedirs(
    output_dir,
    exist_ok=True
)


summary_rows = []


for name, df in [
    ("HC3_train", train_stats),
    ("HC3_validation", validation_stats),
    ("RAID_external", raid_stats)
]:

    summary_rows.append({

        "dataset": name,

        "samples": len(df),

        "avg_characters":
            df["characters"].mean(),

        "median_characters":
            df["characters"].median(),

        "avg_words":
            df["words"].mean(),

        "median_words":
            df["words"].median(),

        "avg_sentences":
            df["sentences"].mean(),

        "avg_vocabulary_diversity":
            df["vocabulary_diversity"].mean()

    })


summary_df = pd.DataFrame(
    summary_rows
)


output_path = os.path.join(
    output_dir,
    "dataset_shift_summary.csv"
)


summary_df.to_csv(
    output_path,
    index=False
)


print()
print("SUMMARY SAVED")
print("=============")

print(
    output_path
)


print()
print(
    "DATASET SHIFT ANALYSIS COMPLETED"
)