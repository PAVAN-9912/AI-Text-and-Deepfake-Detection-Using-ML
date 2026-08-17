import os
import re
import random
import pandas as pd
from collections import defaultdict
from datasets import load_dataset


print()
print("AIGCODESET CODE-AI TRAINING PREPARATION")
print("========================================")


# =========================================================
# SETTINGS
# =========================================================

SEED = 42
random.seed(SEED)

MAX_SCAN = 30000

TRAIN_TOTAL_PER_LABEL = 2000
TEST_TOTAL_PER_LABEL = 500

OUTPUT_DIR = r"datasets\text\CodeAI"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "aigcodeset_training.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "aigcodeset_test.csv"
)


# =========================================================
# LENGTH GROUP
# =========================================================

def word_count(text):

    return len(
        re.findall(
            r"\b\w+\b",
            str(text)
        )
    )


def get_length_group(text):

    n = word_count(text)

    if n <= 15:
        return "very_short"

    elif n <= 50:
        return "short"

    elif n <= 150:
        return "medium"

    else:
        return "long"


# =========================================================
# LOAD DATASET
# =========================================================

print()
print("LOADING AIGCODESET")
print("==================")

print(
    "Streaming mode enabled."
)

print(
    f"Maximum scan: {MAX_SCAN}"
)

print(
    "The full dataset will NOT be downloaded."
)

print()


dataset = load_dataset(
    "basakdemirok/AIGCodeSet",
    split="train",
    streaming=True
)


# =========================================================
# STORAGE
# =========================================================

human_samples = []

ai_samples = []

ai_by_generator = defaultdict(list)

scanned = 0


# =========================================================
# STREAM DATA
# =========================================================

print()
print("COLLECTING CODE SAMPLES")
print("=======================")


for item in dataset:

    scanned += 1

    if scanned % 1000 == 0:

        print(
            f"Scanned {scanned}/{MAX_SCAN}"
        )

    if scanned > MAX_SCAN:
        break


    code = str(
        item.get(
            "code",
            ""
        )
    ).strip()


    if not code:
        continue


    label_raw = str(
        item.get(
            "label",
            ""
        )
    ).strip()


    generator = str(
        item.get(
            "LLM",
            ""
        )
    ).strip()


    # -----------------------------------------------------
    # NORMALIZE LABEL
    # -----------------------------------------------------

    if label_raw == "0":

        label = 0
        label_name = "Human"

    elif label_raw == "1":

        label = 1
        label_name = "AI"

    else:

        continue


    # -----------------------------------------------------
    # LENGTH
    # -----------------------------------------------------

    length_group = get_length_group(
        code
    )


    record = {
        "text": code,
        "label": label,
        "label_name": label_name,
        "generator": generator,
        "length_group": length_group,
        "source": "AIGCodeSet"
    }


    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    if label == 0:

        human_samples.append(
            record
        )

    else:

        ai_samples.append(
            record
        )

        ai_by_generator[
            generator
        ].append(
            record
        )


# =========================================================
# SCAN REPORT
# =========================================================

print()
print("SCAN COMPLETED")
print("===============")

print(
    f"Samples scanned: {scanned}"
)

print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    f"Human: {len(human_samples)}"
)

print(
    f"AI   : {len(ai_samples)}"
)


print()
print("AI GENERATOR DISTRIBUTION")
print("=========================")

for generator in sorted(
    ai_by_generator
):

    print(
        f"{generator:15s}: "
        f"{len(ai_by_generator[generator])}"
    )


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("LENGTH DISTRIBUTION")
print("===================")


all_samples = (
    human_samples
    + ai_samples
)


length_table = (
    pd.DataFrame(all_samples)
    .groupby(
        [
            "length_group",
            "label_name"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print(
    length_table.to_string()
)


# =========================================================
# CHECK MINIMUM DATA
# =========================================================

if len(human_samples) < (
    TRAIN_TOTAL_PER_LABEL
    + TEST_TOTAL_PER_LABEL
):

    raise RuntimeError(
        "Not enough human samples."
    )


if len(ai_samples) < (
    TRAIN_TOTAL_PER_LABEL
    + TEST_TOTAL_PER_LABEL
):

    raise RuntimeError(
        "Not enough AI samples."
    )


# =========================================================
# BUILD BALANCED DATASET
# =========================================================

print()
print("BUILDING BALANCED DATASETS")
print("===========================")


# ---------------------------------------------------------
# Human
# ---------------------------------------------------------

random.shuffle(
    human_samples
)


human_required = (
    TRAIN_TOTAL_PER_LABEL
    + TEST_TOTAL_PER_LABEL
)


selected_human = human_samples[
    :human_required
]


human_train = selected_human[
    :TRAIN_TOTAL_PER_LABEL
]


human_test = selected_human[
    TRAIN_TOTAL_PER_LABEL:
]


# ---------------------------------------------------------
# AI
# ---------------------------------------------------------

# Try to keep AI generators balanced.

generators = [
    g
    for g in [
        "GEMINI",
        "LLAMA",
        "CODESTRAL"
    ]
    if len(
        ai_by_generator[g]
    ) > 0
]


ai_per_generator_train = (
    TRAIN_TOTAL_PER_LABEL
    // len(generators)
)


ai_per_generator_test = (
    TEST_TOTAL_PER_LABEL
    // len(generators)
)


ai_train = []
ai_test = []


for generator in generators:

    pool = list(
        ai_by_generator[
            generator
        ]
    )

    random.shuffle(
        pool
    )


    train_n = min(
        ai_per_generator_train,
        len(pool)
    )


    train_part = pool[
        :train_n
    ]


    remaining = pool[
        train_n:
    ]


    test_n = min(
        ai_per_generator_test,
        len(remaining)
    )


    test_part = remaining[
        :test_n
    ]


    ai_train.extend(
        train_part
    )

    ai_test.extend(
        test_part
    )


# ---------------------------------------------------------
# Fill remaining AI samples if generator balancing leaves
# us short.
# ---------------------------------------------------------

remaining_ai_train = (
    TRAIN_TOTAL_PER_LABEL
    - len(ai_train)
)


remaining_ai_test = (
    TEST_TOTAL_PER_LABEL
    - len(ai_test)
)


used_ids = set(
    id(x)
    for x in (
        ai_train
        + ai_test
    )
)


remaining_pool = [
    x
    for x in ai_samples
    if id(x) not in used_ids
]


random.shuffle(
    remaining_pool
)


if remaining_ai_train > 0:

    extra_train = remaining_pool[
        :remaining_ai_train
    ]

    ai_train.extend(
        extra_train
    )


remaining_pool = [
    x
    for x in remaining_pool
    if id(x)
    not in set(
        id(y)
        for y in ai_train
    )
]


if remaining_ai_test > 0:

    extra_test = remaining_pool[
        :remaining_ai_test
    ]

    ai_test.extend(
        extra_test
    )


# =========================================================
# FINAL TRAIN / TEST
# =========================================================

train_df = pd.DataFrame(
    human_train
    + ai_train
)


test_df = pd.DataFrame(
    human_test
    + ai_test
)


# Shuffle

train_df = train_df.sample(
    frac=1,
    random_state=SEED
).reset_index(
    drop=True
)


test_df = test_df.sample(
    frac=1,
    random_state=SEED + 1
).reset_index(
    drop=True
)


# =========================================================
# CREATE DIRECTORY
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# SAVE
# =========================================================

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)


test_df.to_csv(
    TEST_OUTPUT,
    index=False
)


# =========================================================
# TRAIN REPORT
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
        train_df[
            "length_group"
        ],
        train_df[
            "label_name"
        ]
    ).to_string()
)


print()
print("TRAIN AI GENERATOR DISTRIBUTION")
print("===============================")

print(
    train_df[
        train_df["label"] == 1
    ]["generator"]
    .value_counts()
    .to_string()
)


# =========================================================
# TEST REPORT
# =========================================================

print()
print("TEST DATASET CREATED")
print("====================")

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
        test_df[
            "length_group"
        ],
        test_df[
            "label_name"
        ]
    ).to_string()
)


print()
print("TEST AI GENERATOR DISTRIBUTION")
print("==============================")

print(
    test_df[
        test_df["label"] == 1
    ]["generator"]
    .value_counts()
    .to_string()
)


# =========================================================
# VALIDATION
# =========================================================

print()
print("DATASET VALIDATION")
print("==================")


expected_train = (
    TRAIN_TOTAL_PER_LABEL * 2
)

expected_test = (
    TEST_TOTAL_PER_LABEL * 2
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


if len(train_df) == expected_train:

    print(
        "✓ Training dataset size is correct."
    )

else:

    print(
        "⚠ Training dataset size differs."
    )


if len(test_df) == expected_test:

    print(
        "✓ Test dataset size is correct."
    )

else:

    print(
        "⚠ Test dataset size differs."
    )


print()
print(
    "AIGCODESET PREPARATION COMPLETED"
)