import os
import pandas as pd
import numpy as np


print()
print("RAID FINAL MODEL ERROR ANALYSIS")
print("===============================")
print()


# =========================================================
# SETTINGS
# =========================================================

RESULTS_PATH = (
    r"datasets\text\RAID"
    r"\raid_final_results.csv"
)

OUTPUT_DIR = (
    r"datasets\text\RAID"
)


AI_MISSED_PATH = os.path.join(
    OUTPUT_DIR,
    "raid_ai_missed_errors.csv"
)

HUMAN_FALSE_AI_PATH = os.path.join(
    OUTPUT_DIR,
    "raid_human_false_ai_errors.csv"
)

ERROR_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "raid_error_analysis_summary.csv"
)


# =========================================================
# LOAD RESULTS
# =========================================================

print("LOADING FINAL RAID RESULTS")
print("==========================")

df = pd.read_csv(
    RESULTS_PATH
)

print(
    f"Total samples: {len(df)}"
)


# =========================================================
# NORMALIZE COLUMNS
# =========================================================

if "prediction_name" not in df.columns:

    df["prediction_name"] = (
        df["prediction"]
        .map(
            {
                0: "Human",
                1: "AI"
            }
        )
    )


if "model" not in df.columns:

    if "raid_model" in df.columns:
        df["model"] = df["raid_model"]

    else:
        df["model"] = "unknown"


if "domain" not in df.columns:

    df["domain"] = "unknown"


if "word_count" not in df.columns:

    df["word_count"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )


if "length_group" not in df.columns:

    def get_length_group(n):

        if n <= 10:
            return "very_short"

        elif n <= 30:
            return "short"

        elif n <= 60:
            return "medium"

        else:
            return "long"

    df["length_group"] = (
        df["word_count"]
        .apply(get_length_group)
    )


# =========================================================
# ERROR FLAGS
# =========================================================

df["correct"] = (
    df["label"] ==
    df["prediction"]
)


# AI classified as Human
df["ai_missed"] = (
    (df["label"] == 1) &
    (df["prediction"] == 0)
)


# Human classified as AI
df["human_false_ai"] = (
    (df["label"] == 0) &
    (df["prediction"] == 1)
)


# =========================================================
# BASIC ERROR COUNTS
# =========================================================

ai_missed = df[
    df["ai_missed"]
].copy()


human_false_ai = df[
    df["human_false_ai"]
].copy()


print()
print("ERROR SUMMARY")
print("=============")

print(
    f"Total errors                 : "
    f"{len(df) - df['correct'].sum()}"
)

print(
    f"AI classified as Human      : "
    f"{len(ai_missed)}"
)

print(
    f"Human classified as AI      : "
    f"{len(human_false_ai)}"
)


# =========================================================
# AI MISSED BY GENERATOR
# =========================================================

print()
print("MISSED AI BY GENERATOR")
print("======================")

if len(ai_missed) > 0:

    generator_summary = (
        ai_missed["model"]
        .value_counts()
    )

    print(
        generator_summary
        .to_string()
    )


# =========================================================
# AI MISSED BY DOMAIN
# =========================================================

print()
print("MISSED AI BY DOMAIN")
print("===================")

if len(ai_missed) > 0:

    domain_summary = (
        ai_missed["domain"]
        .value_counts()
    )

    print(
        domain_summary
        .to_string()
    )


# =========================================================
# AI MISSED BY LENGTH
# =========================================================

print()
print("MISSED AI BY TEXT LENGTH")
print("========================")

if len(ai_missed) > 0:

    length_summary = (
        ai_missed["length_group"]
        .value_counts()
    )

    print(
        length_summary
        .to_string()
    )


# =========================================================
# AI MISS RATE BY LENGTH
# =========================================================

print()
print("AI MISS RATE BY TEXT LENGTH")
print("===========================")

ai_only = df[
    df["label"] == 1
].copy()


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = ai_only[
        ai_only["length_group"] == group
    ]

    if len(subset) == 0:
        continue

    missed = (
        subset["prediction"] == 0
    ).sum()

    miss_rate = (
        missed / len(subset)
    )

    print(
        f"{group:12s} "
        f"AI samples={len(subset):4d} "
        f"missed={missed:4d} "
        f"miss rate={miss_rate:.2%}"
    )


# =========================================================
# HUMAN FALSE POSITIVES
# =========================================================

print()
print("HUMAN FALSE AI BY LENGTH")
print("=========================")

if len(human_false_ai) > 0:

    print(
        human_false_ai[
            "length_group"
        ]
        .value_counts()
        .to_string()
    )


# =========================================================
# CONFIDENCE ANALYSIS
# =========================================================

print()
print("AI-MISSED CONFIDENCE")
print("====================")


if (
    len(ai_missed) > 0
    and "ai_probability" in ai_missed.columns
):

    print(
        f"Average AI probability : "
        f"{ai_missed['ai_probability'].mean():.4f}"
    )

    print(
        f"Minimum AI probability : "
        f"{ai_missed['ai_probability'].min():.4f}"
    )

    print(
        f"Maximum AI probability : "
        f"{ai_missed['ai_probability'].max():.4f}"
    )


# =========================================================
# GENERATOR × LENGTH
# =========================================================

print()
print("MISSED AI — GENERATOR × LENGTH")
print("==============================")

if len(ai_missed) > 0:

    table = pd.crosstab(
        ai_missed["model"],
        ai_missed["length_group"]
    )

    print(
        table.to_string()
    )


# =========================================================
# DOMAIN × LENGTH
# =========================================================

print()
print("MISSED AI — DOMAIN × LENGTH")
print("============================")

if len(ai_missed) > 0:

    table = pd.crosstab(
        ai_missed["domain"],
        ai_missed["length_group"]
    )

    print(
        table.to_string()
    )


# =========================================================
# SAVE AI MISSED
# =========================================================

ai_missed.to_csv(
    AI_MISSED_PATH,
    index=False
)


print()
print("AI MISSED FILE SAVED")
print("====================")

print(
    AI_MISSED_PATH
)


# =========================================================
# SAVE HUMAN FALSE AI
# =========================================================

human_false_ai.to_csv(
    HUMAN_FALSE_AI_PATH,
    index=False
)


print()
print("HUMAN FALSE-AI FILE SAVED")
print("==========================")

print(
    HUMAN_FALSE_AI_PATH
)


# =========================================================
# BUILD SUMMARY TABLE
# =========================================================

summary_rows = []


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    subset = ai_only[
        ai_only["length_group"] == group
    ]

    if len(subset) == 0:
        continue

    missed = (
        subset["prediction"] == 0
    ).sum()

    detected = (
        subset["prediction"] == 1
    ).sum()

    summary_rows.append(
        {
            "length_group": group,
            "ai_samples": len(subset),
            "ai_detected": detected,
            "ai_missed": missed,
            "ai_recall":
                detected / len(subset),
            "ai_miss_rate":
                missed / len(subset)
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    ERROR_SUMMARY_PATH,
    index=False
)


print()
print("ERROR SUMMARY SAVED")
print("===================")

print(
    ERROR_SUMMARY_PATH
)


# =========================================================
# SHOW MOST DIFFICULT AI SAMPLES
# =========================================================

print()
print("MOST DIFFICULT AI SAMPLES")
print("==========================")

if len(ai_missed) > 0:

    difficult = ai_missed.sort_values(
        by="ai_probability",
        ascending=True
    )


    show_columns = [
        "model",
        "domain",
        "length_group",
        "word_count",
        "ai_probability",
        "text"
    ]


    available_columns = [
        col
        for col in show_columns
        if col in difficult.columns
    ]


    for index, row in difficult[
        available_columns
    ].head(20).iterrows():

        print()
        print(
            f"Model : {row.get('model', '')}"
        )

        print(
            f"Domain: {row.get('domain', '')}"
        )

        print(
            f"Length: {row.get('length_group', '')}"
        )

        print(
            f"Words : {row.get('word_count', '')}"
        )

        if "ai_probability" in row:

            print(
                f"AI probability: "
                f"{row['ai_probability']:.4f}"
            )

        print(
            "Text:"
        )

        print(
            str(row["text"])[:1000]
        )

        print(
            "-" * 70
        )


# =========================================================
# FINAL
# =========================================================

print()
print("RAID ERROR ANALYSIS COMPLETED")
print("=============================")