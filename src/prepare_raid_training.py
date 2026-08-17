import os
import pandas as pd
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = r"datasets\text\RAID"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "raid_training_sample.csv"
)

TARGET_PER_MODEL = 1000

MODELS = [
    "human",
    "gpt2",
    "llama-chat",
    "mpt",
    "mpt-chat"
]


# ============================================================
# HEADER
# ============================================================

print()
print("RAID TRAINING SAMPLE PREPARATION")
print("================================")


# ============================================================
# LOAD RAID TRAINING DATA
# ============================================================

print()
print("Loading RAID training split in streaming mode...")
print("The full RAID dataset will NOT be downloaded.")

dataset = load_dataset(
    "liamdugan/raid",
    name="raid",
    split="train",
    streaming=True
)


# ============================================================
# STORAGE
# ============================================================

samples = {
    model: []
    for model in MODELS
}


counts = {
    model: 0
    for model in MODELS
}


scanned = 0


# ============================================================
# STREAM DATA
# ============================================================

print()
print("Collecting samples...")
print("=====================")


for item in dataset:

    scanned += 1

    model = str(
        item.get(
            "model",
            ""
        )
    ).strip()


    if model not in MODELS:
        continue


    if counts[model] >= TARGET_PER_MODEL:
        continue


    text = item.get(
        "generation"
    )


    if text is None:
        continue


    text = str(
        text
    ).strip()


    if len(text) < 20:
        continue


    samples[model].append({

        "text": text,

        "label": (
            0
            if model == "human"
            else 1
        ),

        "source": "raid",

        "raid_model": model,

        "domain": item.get(
            "domain"
        ),

        "attack": item.get(
            "attack"
        )

    })


    counts[model] += 1


    if scanned % 1000 == 0:

        print(
            f"Scanned: {scanned} | "
            + " | ".join(
                f"{m}: {counts[m]}/{TARGET_PER_MODEL}"
                for m in MODELS
            )
        )


    if all(
        counts[m] >= TARGET_PER_MODEL
        for m in MODELS
    ):

        break


# ============================================================
# SUMMARY
# ============================================================

print()
print("COLLECTION COMPLETED")
print("====================")

for model in MODELS:

    print(
        f"{model:12s}: {len(samples[model])}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

rows = []

for model in MODELS:

    rows.extend(
        samples[model]
    )


df = pd.DataFrame(
    rows
)


# ============================================================
# SHUFFLE
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print()
print("RAID TRAINING DATASET CREATED")
print("=============================")

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Total samples:",
    len(df)
)


print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label"].value_counts()
)


print()
print("MODEL DISTRIBUTION")
print("==================")

print(
    df["raid_model"].value_counts()
)


print()
print("DOMAIN DISTRIBUTION")
print("===================")

print(
    df["domain"].value_counts()
)


print()
print("DATASET READY")
print("==============")


print(
    "IMPORTANT:"
)

print(
    "raid_external.csv remains untouched "
    "for external evaluation."
)