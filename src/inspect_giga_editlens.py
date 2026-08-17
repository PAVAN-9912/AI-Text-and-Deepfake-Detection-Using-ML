from datasets import load_dataset
from collections import Counter


print()
print("GIGA-EDITLENS DEEP INSPECTION")
print("=============================")

print()
print("Loading Giga-EditLens in streaming mode...")
print("The full dataset will NOT be downloaded.")


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TARGET_HUMAN = 1000
TARGET_AI = 1000
TARGET_AI_EDITED = 500

MAX_SCANNED = 300000


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

dataset = load_dataset(
    "rounaksaha12/giga-editlens",
    split="train",
    streaming=True
)


# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

label_counts = Counter()

length_counts = Counter()

length_label_counts = Counter()

model_counts = Counter()

source_counts = Counter()


# ---------------------------------------------------------
# LENGTH FUNCTION
# ---------------------------------------------------------

def length_group(word_count):

    if word_count <= 30:

        return "short"

    elif word_count <= 100:

        return "medium"

    elif word_count <= 250:

        return "long"

    else:

        return "very_long"


# ---------------------------------------------------------
# SCAN
# ---------------------------------------------------------

print()
print(
    f"Scanning up to {MAX_SCANNED} samples..."
)

print(
    "The scan will stop early when enough examples are found."
)

print()


scanned = 0


for item in dataset:

    scanned += 1


    if scanned > MAX_SCANNED:

        break


    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    text = str(
        item.get(
            "text",
            ""
        )
    ).strip()


    if not text:

        continue


    word_count = len(
        text.split()
    )


    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    text_type = str(
        item.get(
            "text_type",
            ""
        )
    ).strip()


    if text_type == "human_written":

        label = "Human"

    elif text_type == "ai_generated":

        label = "AI"

    elif text_type == "ai_edited":

        label = "AI_Edited"

    else:

        label = "Unknown"


    # -----------------------------------------------------
    # LENGTH
    # -----------------------------------------------------

    group = length_group(
        word_count
    )


    # -----------------------------------------------------
    # COUNTERS
    # -----------------------------------------------------

    label_counts[label] += 1

    length_counts[group] += 1

    length_label_counts[
        (group, label)
    ] += 1


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = str(
        item.get(
            "model",
            ""
        )
    ).strip()


    if model:

        model_counts[model] += 1


    # -----------------------------------------------------
    # SOURCE
    # -----------------------------------------------------

    source = str(
        item.get(
            "source",
            ""
        )
    ).strip()


    if source:

        source_counts[source] += 1


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if scanned % 5000 == 0:

        print(
            f"Scanned {scanned:,} | "
            f"Human={label_counts['Human']} | "
            f"AI={label_counts['AI']} | "
            f"AI-Edited={label_counts['AI_Edited']}"
        )


    # -----------------------------------------------------
    # STOP CONDITION
    # -----------------------------------------------------

    if (
        label_counts["Human"] >= TARGET_HUMAN
        and
        label_counts["AI"] >= TARGET_AI
        and
        label_counts["AI_Edited"] >= TARGET_AI_EDITED
    ):

        break


# ---------------------------------------------------------
# FINAL SCAN INFORMATION
# ---------------------------------------------------------

print()
print("SCAN COMPLETED")
print("===============")

print(
    f"Total records scanned: {scanned:,}"
)


# ---------------------------------------------------------
# LABEL DISTRIBUTION
# ---------------------------------------------------------

print()
print("LABEL DISTRIBUTION")
print("==================")

for label in [
    "Human",
    "AI",
    "AI_Edited",
    "Unknown"
]:

    print(
        f"{label:12s}: "
        f"{label_counts[label]}"
    )


# ---------------------------------------------------------
# LENGTH DISTRIBUTION
# ---------------------------------------------------------

print()
print("OVERALL LENGTH DISTRIBUTION")
print("===========================")

for group in [
    "short",
    "medium",
    "long",
    "very_long"
]:

    print(
        f"{group:12s}: "
        f"{length_counts[group]}"
    )


# ---------------------------------------------------------
# LENGTH × LABEL
# ---------------------------------------------------------

print()
print("LENGTH × LABEL DISTRIBUTION")
print("===========================")

print(
    f"{'Length':12s}"
    f"{'Human':12s}"
    f"{'AI':12s}"
    f"{'AI_Edited':12s}"
)


for group in [
    "short",
    "medium",
    "long",
    "very_long"
]:

    human = length_label_counts[
        (group, "Human")
    ]

    ai = length_label_counts[
        (group, "AI")
    ]

    edited = length_label_counts[
        (group, "AI_Edited")
    ]


    print(
        f"{group:12s}"
        f"{human:<12d}"
        f"{ai:<12d}"
        f"{edited:<12d}"
    )


# ---------------------------------------------------------
# MODEL DISTRIBUTION
# ---------------------------------------------------------

print()
print("MODEL DISTRIBUTION")
print("==================")

for model, count in (
    model_counts.most_common()
):

    print(
        f"{model:40s}: {count}"
    )


# ---------------------------------------------------------
# SOURCE DISTRIBUTION
# ---------------------------------------------------------

print()
print("SOURCE DISTRIBUTION")
print("===================")

for source, count in (
    source_counts.most_common()
):

    print(
        f"{source:35s}: {count}"
    )


# ---------------------------------------------------------
# SHORT TEXT SUMMARY
# ---------------------------------------------------------

print()
print("SHORT TEXT SUMMARY")
print("==================")

for label in [
    "Human",
    "AI",
    "AI_Edited"
]:

    count = length_label_counts[
        ("short", label)
    ]

    print(
        f"{label:12s}: "
        f"{count}"
    )


# ---------------------------------------------------------
# MEDIUM TEXT SUMMARY
# ---------------------------------------------------------

print()
print("MEDIUM TEXT SUMMARY")
print("===================")

for label in [
    "Human",
    "AI",
    "AI_Edited"
]:

    count = length_label_counts[
        ("medium", label)
    ]

    print(
        f"{label:12s}: "
        f"{count}"
    )


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print()
print(
    "GIGA-EDITLENS DEEP INSPECTION COMPLETED"
)