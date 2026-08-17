from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(r"datasets\text\RAID")
OUTPUT_FILE = OUTPUT_DIR / "raid_external.csv"

HUMAN_TARGET = 500

AI_TARGETS = {
    "llama-chat": 250,
    "mpt": 250,
    "mpt-chat": 250,
    "gpt2": 250,
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RAID IN STREAMING MODE
# ============================================================

print()
print("RAID EXTERNAL BENCHMARK")
print("=======================")

print()
print("Loading RAID extra split in streaming mode...")
print("The full RAID dataset will NOT be downloaded.")


dataset = load_dataset(
    "liamdugan/raid",
    split="extra",
    streaming=True
)


# ============================================================
# STORAGE
# ============================================================

human_samples = []

ai_samples = defaultdict(list)


# ============================================================
# REQUIRED AI MODELS
# ============================================================

required_ai_models = set(
    AI_TARGETS.keys()
)


# ============================================================
# STREAM DATA
# ============================================================

print()
print("Collecting samples...")
print("=====================")

scanned = 0

for row in dataset:

    scanned += 1

    model_name = str(
        row.get("model", "")
    ).strip().lower()

    generation = row.get(
        "generation"
    )

    # --------------------------------------------------------
    # Ignore empty generations
    # --------------------------------------------------------

    if generation is None:
        continue

    generation = str(
        generation
    ).strip()

    if not generation:
        continue

    # --------------------------------------------------------
    # HUMAN
    # --------------------------------------------------------

    if model_name == "human":

        if len(human_samples) < HUMAN_TARGET:

            human_samples.append(
                {
                    "text": generation,
                    "label": 0,
                    "label_name": "Human",
                    "model": model_name,
                    "domain": row.get("domain"),
                    "attack": row.get("attack"),
                    "decoding": row.get("decoding"),
                    "repetition_penalty":
                        row.get("repetition_penalty"),
                    "title": row.get("title"),
                    "prompt": row.get("prompt"),
                    "source_id": row.get("source_id"),
                    "raid_id": row.get("id"),
                }
            )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    elif model_name in required_ai_models:

        target = AI_TARGETS[
            model_name
        ]

        if len(
            ai_samples[model_name]
        ) < target:

            ai_samples[model_name].append(
                {
                    "text": generation,
                    "label": 1,
                    "label_name": "AI",
                    "model": model_name,
                    "domain": row.get("domain"),
                    "attack": row.get("attack"),
                    "decoding": row.get("decoding"),
                    "repetition_penalty":
                        row.get("repetition_penalty"),
                    "title": row.get("title"),
                    "prompt": row.get("prompt"),
                    "source_id": row.get("source_id"),
                    "raid_id": row.get("id"),
                }
            )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if scanned % 1000 == 0:

        collected_ai = sum(
            len(v)
            for v in ai_samples.values()
        )

        print(
            f"Scanned: {scanned} | "
            f"Human: {len(human_samples)}/{HUMAN_TARGET} | "
            f"AI: {collected_ai}/1000"
        )

    # --------------------------------------------------------
    # Stop when everything is collected
    # --------------------------------------------------------

    human_done = (
        len(human_samples)
        >= HUMAN_TARGET
    )

    ai_done = all(
        len(ai_samples[model])
        >= AI_TARGETS[model]
        for model in AI_TARGETS
    )

    if human_done and ai_done:

        break


# ============================================================
# VERIFY COLLECTION
# ============================================================

print()
print("COLLECTION COMPLETED")
print("====================")

print(
    "Human samples:",
    len(human_samples)
)

for model_name in AI_TARGETS:

    print(
        f"{model_name}:",
        len(ai_samples[model_name])
    )


# ============================================================
# COMBINE DATA
# ============================================================

records = []

records.extend(
    human_samples
)

for model_name in AI_TARGETS:

    records.extend(
        ai_samples[model_name]
    )


df = pd.DataFrame(
    records
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
# ADD SAMPLE ID
# ============================================================

df.insert(
    0,
    "sample_id",
    range(len(df))
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("RAID EXTERNAL DATASET CREATED")
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
    df["label_name"].value_counts()
)

print()
print("MODEL DISTRIBUTION")
print("==================")

print(
    df["model"].value_counts()
)

print()
print("DOMAIN DISTRIBUTION")
print("===================")

print(
    df["domain"].value_counts()
)

print()
print("DATASET READY FOR EXTERNAL EVALUATION")