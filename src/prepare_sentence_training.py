import os
import random

import pandas as pd
from collections import defaultdict
from datasets import load_dataset


print()
print("SHORT-TEXT AI DETECTION DATA PREPARATION")
print("========================================")


# =========================================================
# SETTINGS
# =========================================================

DATASET_NAME = "shahxeebhassan/human_vs_ai_sentences"

MAX_SCAN = 20000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# Training targets
TRAIN_TARGETS = {
    "very_short": 1000,
    "short": 1500,
    "medium": 500
}


# Test targets
TEST_TARGETS = {
    "very_short": 300,
    "short": 500,
    "medium": 200
}


OUTPUT_DIR = r"datasets\text\SentenceAI"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sentence_ai_training.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sentence_ai_test.csv"
)


# =========================================================
# LENGTH FUNCTION
# =========================================================

def get_length_group(word_count):

    if word_count <= 10:
        return "very_short"

    elif word_count <= 30:
        return "short"

    elif word_count <= 60:
        return "medium"

    else:
        return "long"


# =========================================================
# LOAD DATASET
# =========================================================

print()
print("LOADING SENTENCE DATASET")
print("========================")

print(
    f"Dataset: {DATASET_NAME}"
)

print(
    "Streaming mode enabled."
)

print(
    f"Maximum scan: {MAX_SCAN} samples."
)


dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)


# =========================================================
# STORAGE
# =========================================================

candidates = defaultdict(list)


# =========================================================
# COLLECT CANDIDATES
# =========================================================

print()
print("COLLECTING CANDIDATES")
print("=====================")


scanned = 0


for item in dataset:

    if scanned >= MAX_SCAN:
        break


    text = str(
        item.get(
            "text",
            ""
        )
    ).strip()


    if not text:
        scanned += 1
        continue


    raw_label = item.get(
        "label",
        None
    )


    if raw_label not in [0, 1]:
        scanned += 1
        continue


    label = int(raw_label)


    word_count = len(
        text.split()
    )


    length_group = get_length_group(
        word_count
    )


    # We only need short and medium text.
    if length_group not in [
        "very_short",
        "short",
        "medium"
    ]:

        scanned += 1
        continue


    candidates[
        (
            length_group,
            label
        )
    ].append(
        {
            "text": text,
            "label": label,
            "label_name": (
                "Human"
                if label == 0
                else "AI"
            ),
            "word_count": word_count,
            "length_group": length_group
        }
    )


    scanned += 1


    if scanned % 1000 == 0:

        print(
            f"Scanned {scanned}/{MAX_SCAN}"
        )


# =========================================================
# CANDIDATE REPORT
# =========================================================

print()
print("CANDIDATE DISTRIBUTION")
print("======================")

print(
    f"{'Length':12s}"
    f"{'Human':12s}"
    f"{'AI':12s}"
)


for group in [
    "very_short",
    "short",
    "medium"
]:

    human_count = len(
        candidates[
            (group, 0)
        ]
    )

    ai_count = len(
        candidates[
            (group, 1)
        ]
    )


    print(
        f"{group:12s}"
        f"{human_count:<12d}"
        f"{ai_count:<12d}"
    )


# =========================================================
# CHECK TARGET AVAILABILITY
# =========================================================

print()
print("CHECKING TARGET AVAILABILITY")
print("============================")


for group in TRAIN_TARGETS:

    required = (
        TRAIN_TARGETS[group]
        +
        TEST_TARGETS[group]
    )


    for label in [0, 1]:

        available = len(
            candidates[
                (group, label)
            ]
        )


        label_name = (
            "Human"
            if label == 0
            else "AI"
        )


        print(
            f"{group:12s} "
            f"{label_name:6s} "
            f"Available={available:5d} "
            f"Required={required:5d}"
        )


        if available < required:

            raise RuntimeError(
                f"Not enough {label_name} "
                f"samples for {group}. "
                f"Required={required}, "
                f"Available={available}"
            )


# =========================================================
# BUILD TRAINING + TEST
# =========================================================

train_rows = []

test_rows = []


print()
print("BUILDING BALANCED DATASETS")
print("===========================")


for group in [
    "very_short",
    "short",
    "medium"
]:

    for label in [0, 1]:

        pool = list(
            candidates[
                (group, label)
            ]
        )


        random.shuffle(
            pool
        )


        train_count = TRAIN_TARGETS[
            group
        ]

        test_count = TEST_TARGETS[
            group
        ]


        selected_train = pool[
            :train_count
        ]

        selected_test = pool[
            train_count:
            train_count + test_count
        ]


        train_rows.extend(
            selected_train
        )

        test_rows.extend(
            selected_test
        )


        label_name = (
            "Human"
            if label == 0
            else "AI"
        )


        print(
            f"{group:12s} "
            f"{label_name:6s} "
            f"Train={len(selected_train):4d} "
            f"Test={len(selected_test):4d}"
        )


# =========================================================
# SHUFFLE
# =========================================================

random.shuffle(
    train_rows
)

random.shuffle(
    test_rows
)


# =========================================================
# DATAFRAMES
# =========================================================

train_df = pd.DataFrame(
    train_rows
)

test_df = pd.DataFrame(
    test_rows
)


# =========================================================
# ADD SOURCE
# =========================================================

train_df["source"] = (
    "human_vs_ai_sentences"
)

test_df["source"] = (
    "human_vs_ai_sentences"
)


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# SAVE TRAINING DATA
# =========================================================

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)


# =========================================================
# SAVE TEST DATA
# =========================================================

test_df.to_csv(
    TEST_OUTPUT,
    index=False
)


# =========================================================
# TRAINING REPORT
# =========================================================

print()
print("TRAINING DATASET CREATED")
print("========================")

print(
    f"Output: {TRAIN_OUTPUT}"
)

print(
    f"Total samples: {len(train_df)}"
)


print()
print("TRAIN LABEL DISTRIBUTION")
print("========================")

print(
    train_df[
        "label_name"
    ]
    .value_counts()
    .to_string()
)


print()
print("TRAIN LENGTH DISTRIBUTION")
print("=========================")

print(
    pd.crosstab(
        train_df["length_group"],
        train_df["label_name"]
    )
    .to_string()
)


# =========================================================
# TEST REPORT
# =========================================================

print()
print("SHORT-TEXT TEST DATASET CREATED")
print("===============================")

print(
    f"Output: {TEST_OUTPUT}"
)

print(
    f"Total samples: {len(test_df)}"
)


print()
print("TEST LABEL DISTRIBUTION")
print("=======================")

print(
    test_df[
        "label_name"
    ]
    .value_counts()
    .to_string()
)


print()
print("TEST LENGTH DISTRIBUTION")
print("========================")

print(
    pd.crosstab(
        test_df["length_group"],
        test_df["label_name"]
    )
    .to_string()
)


# =========================================================
# FINAL VALIDATION
# =========================================================

print()
print("DATASET VALIDATION")
print("==================")


expected_train = (
    2 * (
        TRAIN_TARGETS["very_short"]
        +
        TRAIN_TARGETS["short"]
        +
        TRAIN_TARGETS["medium"]
    )
)


expected_test = (
    2 * (
        TEST_TARGETS["very_short"]
        +
        TEST_TARGETS["short"]
        +
        TEST_TARGETS["medium"]
    )
)


print(
    f"Expected training samples: "
    f"{expected_train}"
)

print(
    f"Actual training samples  : "
    f"{len(train_df)}"
)


print(
    f"Expected test samples: "
    f"{expected_test}"
)

print(
    f"Actual test samples  : "
    f"{len(test_df)}"
)


if len(train_df) != expected_train:

    raise RuntimeError(
        "Training dataset size mismatch."
    )


if len(test_df) != expected_test:

    raise RuntimeError(
        "Test dataset size mismatch."
    )


# =========================================================
# EXACT BALANCE CHECK
# =========================================================

train_counts = pd.crosstab(
    train_df["length_group"],
    train_df["label"]
)


test_counts = pd.crosstab(
    test_df["length_group"],
    test_df["label"]
)


for group in [
    "very_short",
    "short",
    "medium"
]:

    train_human = train_counts.loc[
        group, 0
    ]

    train_ai = train_counts.loc[
        group, 1
    ]


    test_human = test_counts.loc[
        group, 0
    ]

    test_ai = test_counts.loc[
        group, 1
    ]


    if train_human != train_ai:

        raise RuntimeError(
            f"Training imbalance detected "
            f"in {group}."
        )


    if test_human != test_ai:

        raise RuntimeError(
            f"Test imbalance detected "
            f"in {group}."
        )


print()
print(
    "✓ Training dataset size is correct."
)

print(
    "✓ Test dataset size is correct."
)

print(
    "✓ Every selected length group is label-balanced."
)


print()
print(
    "SHORT-TEXT DATA PREPARATION COMPLETED"
)