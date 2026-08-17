import pandas as pd
from collections import defaultdict

from datasets import load_dataset


print()
print("RAID DOMAIN-BALANCED TRAINING PREPARATION")
print("=========================================")


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TARGET_PER_MODEL_DOMAIN = 500

TARGET_DOMAINS = [
    "code",
    "german"
]

TARGET_MODELS = [
    "human",
    "gpt2",
    "llama-chat",
    "mpt",
    "mpt-chat"
]


OUTPUT_PATH = (
    "datasets\\text\\RAID\\raid_domain_training_sample.csv"
)


# ---------------------------------------------------------
# LOAD RAID EXTRA
# ---------------------------------------------------------

print()
print("Loading RAID extra split in streaming mode...")
print("The full RAID dataset will NOT be downloaded.")
print()


dataset = load_dataset(
    "liamdugan/raid",
    name="raid",
    split="extra",
    streaming=True
)


# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

samples = defaultdict(list)

counts = defaultdict(int)


print()
print("Collecting balanced code + german samples...")
print("============================================")
print()


# ---------------------------------------------------------
# STREAM RAID DATA
# ---------------------------------------------------------

for item in dataset:

    model = str(
        item.get("model", "")
    ).strip()

    domain = str(
        item.get("domain", "")
    ).strip()


    # Only required models
    if model not in TARGET_MODELS:
        continue


    # Only required domains
    if domain not in TARGET_DOMAINS:
        continue


    key = (model, domain)


    # Already enough samples for this
    # model/domain combination
    if counts[key] >= TARGET_PER_MODEL_DOMAIN:
        continue


    generation = item.get(
        "generation",
        ""
    )


    if not generation:
        continue


    # Store sample
    samples[model].append(
        {
            "text": str(generation),
            "raid_model": model,
            "domain": domain
        }
    )


    counts[key] += 1


    # -----------------------------------------------------
    # STOP CONDITION
    # -----------------------------------------------------
    # Stop only when EVERY model has 500 code
    # AND 500 german samples.
    # -----------------------------------------------------

    if all(
        counts[(model, domain)]
        >= TARGET_PER_MODEL_DOMAIN

        for model in TARGET_MODELS

        for domain in TARGET_DOMAINS
    ):
        break


# ---------------------------------------------------------
# REPORT COLLECTION
# ---------------------------------------------------------

print()
print("COLLECTION COMPLETED")
print("====================")
print()


for model in TARGET_MODELS:

    for domain in TARGET_DOMAINS:

        print(
            f"{model:15s} "
            f"{domain:10s}: "
            f"{counts[(model, domain)]}"
        )


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

rows = []


for model in TARGET_MODELS:

    rows.extend(
        samples[model]
    )


df = pd.DataFrame(rows)


# ---------------------------------------------------------
# CREATE LABEL
# ---------------------------------------------------------

df["label"] = (
    df["raid_model"]
    .apply(
        lambda x:
        0 if x == "human" else 1
    )
)


df["label_name"] = (
    df["label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("RAID DOMAIN TRAINING DATASET CREATED")
print("====================================")
print()


print(
    f"Output: {OUTPUT_PATH}"
)


print(
    f"Total samples: {len(df)}"
)


# ---------------------------------------------------------
# LABEL DISTRIBUTION
# ---------------------------------------------------------

print()
print("LABEL DISTRIBUTION")
print("==================")
print()


print(
    df["label_name"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# MODEL DISTRIBUTION
# ---------------------------------------------------------

print()
print("MODEL DISTRIBUTION")
print("==================")
print()


print(
    df["raid_model"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# DOMAIN DISTRIBUTION
# ---------------------------------------------------------

print()
print("DOMAIN DISTRIBUTION")
print("===================")
print()


print(
    df["domain"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# MODEL × DOMAIN DISTRIBUTION
# ---------------------------------------------------------

print()
print("MODEL × DOMAIN DISTRIBUTION")
print("============================")
print()


model_domain_table = pd.crosstab(
    df["raid_model"],
    df["domain"]
)


print(
    model_domain_table
    .to_string()
)


# ---------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------

print()
print("DATASET VALIDATION")
print("==================")
print()


expected_total = (
    len(TARGET_MODELS)
    * len(TARGET_DOMAINS)
    * TARGET_PER_MODEL_DOMAIN
)


print(
    f"Expected samples: {expected_total}"
)


print(
    f"Actual samples  : {len(df)}"
)


if len(df) == expected_total:

    print()
    print("✓ Dataset size is correct.")


else:

    print()
    print(
        "WARNING: Dataset size is different "
        "from expected."
    )


# Check every model/domain combination
all_balanced = True


for model in TARGET_MODELS:

    for domain in TARGET_DOMAINS:

        actual = counts[
            (model, domain)
        ]

        if actual != TARGET_PER_MODEL_DOMAIN:

            all_balanced = False

            print(
                f"WARNING: {model} / {domain} "
                f"= {actual}"
            )


if all_balanced:

    print(
        "✓ Every model/domain combination "
        "has exactly 500 samples."
    )


# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print()
print("RAID DOMAIN-BALANCED DATASET READY")
print("===================================")
print()