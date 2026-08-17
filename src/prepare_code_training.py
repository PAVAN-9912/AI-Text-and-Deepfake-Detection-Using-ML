import os
import re
import random
import pandas as pd
from collections import defaultdict
from datasets import load_dataset


print()
print("CODE-AI TRAINING DATA PREPARATION")
print("=================================")


# =========================================================
# SETTINGS
# =========================================================

SEED = 42
random.seed(SEED)

MAX_SCAN = 30000

TRAIN_PER_GROUP = {
    "very_short": 1000,
    "short": 1500,
    "medium": 1000,
    "long": 500
}

TEST_PER_GROUP = {
    "very_short": 250,
    "short": 500,
    "medium": 300,
    "long": 150
}

TOTAL_PER_LABEL = {
    "train": sum(TRAIN_PER_GROUP.values()),
    "test": sum(TEST_PER_GROUP.values())
}

OUTPUT_DIR = r"datasets\text\CodeAI"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "code_ai_training.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "code_ai_test.csv"
)


# =========================================================
# LENGTH CLASSIFICATION
# =========================================================

def count_words(text):
    return len(
        re.findall(
            r"\b\w+\b",
            text
        )
    )


def classify_length(text):

    words = count_words(text)

    if words <= 15:
        return "very_short"

    if words <= 50:
        return "short"

    if words <= 150:
        return "medium"

    return "long"


# =========================================================
# CODE DETECTION
# =========================================================

CODE_PATTERNS = [

    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",

    r"#include\s*[<\"]",
    r"\bpublic\s+static\s+void\s+main\b",
    r"\bpublic\s+class\s+\w+",

    r"\bfunction\s+\w+\s*\(",
    r"\bconst\s+\w+\s*=",
    r"\blet\s+\w+\s*=",
    r"\bvar\s+\w+\s*=",

    r"\bif\s*\(",
    r"\bfor\s*\(",
    r"\bwhile\s*\(",

    r"=>",
    r"::",
    r"\{\s*\}",
    r";\s*$",

    r"\breturn\s+",
    r"\bprint\s*\(",
    r"\bconsole\.log\s*\(",

]


def looks_like_code(text):

    score = 0

    for pattern in CODE_PATTERNS:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            score += 1

    # Strong programming indicators
    if "```" in text:
        score += 3

    if re.search(
        r"\b(def|class|return|import|#include)\b",
        text
    ):
        score += 2

    return score >= 1


# =========================================================
# LOAD RAID EXTRA
# =========================================================

print()
print("LOADING RAID EXTRA")
print("==================")
print("Streaming mode enabled.")
print(
    f"Maximum scan: {MAX_SCAN} samples."
)
print()


try:

    dataset = load_dataset(
        "liamdugan/raid",
        name="raid",
        split="extra",
        streaming=True
    )

except Exception as e:

    print()
    print("ERROR LOADING RAID")
    print("==================")
    print(e)
    raise SystemExit(1)


# =========================================================
# STORAGE
# =========================================================

candidates = defaultdict(
    list
)

scanned = 0


print()
print("COLLECTING CODE SAMPLES")
print("=======================")


# =========================================================
# STREAM DATA
# =========================================================

for item in dataset:

    scanned += 1

    if scanned % 1000 == 0:

        print(
            f"Scanned {scanned}/{MAX_SCAN}"
        )

    if scanned > MAX_SCAN:
        break


    model = str(
        item.get(
            "model",
            ""
        )
    ).strip().lower()


    generation = item.get(
        "generation",
        ""
    )


    if not generation:
        continue


    text = str(
        generation
    ).strip()


    if len(text) < 5:
        continue


    # -----------------------------------------------------
    # Keep only target generators
    # -----------------------------------------------------

    if model not in {
        "human",
        "gpt2",
        "llama-chat",
        "mpt",
        "mpt-chat"
    }:
        continue


    # -----------------------------------------------------
    # Keep code only
    # -----------------------------------------------------

    if not looks_like_code(text):
        continue


    # -----------------------------------------------------
    # Label
    # -----------------------------------------------------

    if model == "human":

        label = 0
        label_name = "Human"

    else:

        label = 1
        label_name = "AI"


    # -----------------------------------------------------
    # Length
    # -----------------------------------------------------

    length_group = classify_length(
        text
    )


    candidates[
        (label_name, length_group)
    ].append(
        {
            "text": text,
            "label": label,
            "label_name": label_name,
            "raid_model": model,
            "length_group": length_group,
            "domain": "code"
        }
    )


# =========================================================
# CANDIDATE REPORT
# =========================================================

print()
print("CODE CANDIDATE DISTRIBUTION")
print("===========================")


candidate_rows = []

for key, values in sorted(
    candidates.items()
):

    label_name, length_group = key

    candidate_rows.append(
        {
            "label_name": label_name,
            "length_group": length_group,
            "count": len(values)
        }
    )

    print(
        f"{length_group:12s} "
        f"{label_name:6s} "
        f"{len(values):5d}"
    )


candidate_df = pd.DataFrame(
    candidate_rows
)


# =========================================================
# CHECK AVAILABILITY
# =========================================================

print()
print("CHECKING TARGET AVAILABILITY")
print("============================")


required = {
    "Human": TRAIN_PER_GROUP,
    "AI": TRAIN_PER_GROUP
}

required_test = {
    "Human": TEST_PER_GROUP,
    "AI": TEST_PER_GROUP
}


for label_name in [
    "Human",
    "AI"
]:

    for length_group in [
        "very_short",
        "short",
        "medium",
        "long"
    ]:

        available = len(
            candidates[
                (
                    label_name,
                    length_group
                )
            ]
        )

        train_required = TRAIN_PER_GROUP[
            length_group
        ]

        test_required = TEST_PER_GROUP[
            length_group
        ]

        total_required = (
            train_required
            + test_required
        )

        print(
            f"{length_group:12s} "
            f"{label_name:6s} "
            f"Available={available:5d} "
            f"Required={total_required:5d}"
        )

        if available < total_required:

            print()
            print(
                "WARNING: Not enough samples."
            )
            print(
                f"{label_name} / "
                f"{length_group}"
            )
            print(
                f"Available: {available}"
            )
            print(
                f"Required : {total_required}"
            )


# =========================================================
# BUILD TRAIN + TEST
# =========================================================

print()
print("BUILDING CODE TRAINING DATA")
print("===========================")


train_rows = []
test_rows = []


for label_name in [
    "Human",
    "AI"
]:

    for length_group in [
        "very_short",
        "short",
        "medium",
        "long"
    ]:

        pool = list(
            candidates[
                (
                    label_name,
                    length_group
                )
            ]
        )


        random.shuffle(
            pool
        )


        train_required = TRAIN_PER_GROUP[
            length_group
        ]

        test_required = TEST_PER_GROUP[
            length_group
        ]


        total_required = (
            train_required
            + test_required
        )


        selected = pool[
            :total_required
        ]


        train_part = selected[
            :train_required
        ]

        test_part = selected[
            train_required:
            train_required
            + test_required
        ]


        train_rows.extend(
            train_part
        )

        test_rows.extend(
            test_part
        )


        print(
            f"{length_group:12s} "
            f"{label_name:6s} "
            f"Train={len(train_part):4d} "
            f"Test={len(test_part):4d}"
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
# SHUFFLE
# =========================================================

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
print("CODE TRAINING DATASET CREATED")
print("=============================")

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
    ].value_counts().to_string()
)


print()
print("TRAIN LENGTH DISTRIBUTION")
print("=========================")

print(
    pd.crosstab(
        train_df["length_group"],
        train_df["label_name"]
    ).to_string()
)


print()
print("TRAIN MODEL DISTRIBUTION")
print("========================")

print(
    train_df[
        "raid_model"
    ].value_counts().to_string()
)


# =========================================================
# TEST REPORT
# =========================================================

print()
print("CODE TEST DATASET CREATED")
print("==========================")

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
    ].value_counts().to_string()
)


print()
print("TEST LENGTH DISTRIBUTION")
print("========================")

print(
    pd.crosstab(
        test_df["length_group"],
        test_df["label_name"]
    ).to_string()
)


# =========================================================
# VALIDATION
# =========================================================

print()
print("DATASET VALIDATION")
print("==================")


expected_train = (
    2
    * sum(
        TRAIN_PER_GROUP.values()
    )
)

expected_test = (
    2
    * sum(
        TEST_PER_GROUP.values()
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


# =========================================================
# FINAL
# =========================================================

print()
print(
    "CODE-AI DATA PREPARATION COMPLETED"
)