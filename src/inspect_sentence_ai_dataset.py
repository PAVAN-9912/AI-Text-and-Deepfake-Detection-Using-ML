from datasets import load_dataset
from collections import Counter


print()
print("HUMAN VS AI SENTENCE DATASET INSPECTION")
print("=======================================")

print()
print("Dataset:")
print("shahxeebhassan/human_vs_ai_sentences")
print()
print("Loading in streaming mode...")
print("The full dataset will NOT be downloaded.")


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

dataset = load_dataset(
    "shahxeebhassan/human_vs_ai_sentences",
    split="train",
    streaming=True
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MAX_SAMPLES = 20000


# ---------------------------------------------------------
# COUNTERS
# ---------------------------------------------------------

label_counts = Counter()
length_counts = Counter()
length_label_counts = Counter()


# ---------------------------------------------------------
# LENGTH GROUP
# ---------------------------------------------------------

def get_length_group(word_count):

    if word_count <= 10:
        return "very_short"

    elif word_count <= 30:
        return "short"

    elif word_count <= 60:
        return "medium"

    else:
        return "long"


# ---------------------------------------------------------
# SCAN
# ---------------------------------------------------------

print()
print(
    f"Scanning first {MAX_SAMPLES} samples..."
)

print()


scanned = 0


for item in dataset:

    if scanned >= MAX_SAMPLES:
        break


    text = str(
        item.get(
            "text",
            ""
        )
    ).strip()


    if not text:
        continue


    label = item.get(
        "label",
        None
    )


    if label == 0:

        label_name = "Human"

    elif label == 1:

        label_name = "AI"

    else:

        label_name = "Unknown"


    word_count = len(
        text.split()
    )


    length_group = get_length_group(
        word_count
    )


    label_counts[label_name] += 1

    length_counts[length_group] += 1

    length_label_counts[
        (length_group, label_name)
    ] += 1


    scanned += 1


    if scanned % 1000 == 0:

        print(
            f"Scanned {scanned}/{MAX_SAMPLES}"
        )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("SCAN COMPLETED")
print("===============")

print(
    f"Samples scanned: {scanned}"
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
    "Unknown"
]:

    print(
        f"{label:10s}: "
        f"{label_counts[label]}"
    )


# ---------------------------------------------------------
# LENGTH DISTRIBUTION
# ---------------------------------------------------------

print()
print("LENGTH DISTRIBUTION")
print("===================")

for group in [
    "very_short",
    "short",
    "medium",
    "long"
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
)


for group in [
    "very_short",
    "short",
    "medium",
    "long"
]:

    human = length_label_counts[
        (group, "Human")
    ]

    ai = length_label_counts[
        (group, "AI")
    ]


    print(
        f"{group:12s}"
        f"{human:<12d}"
        f"{ai:<12d}"
    )


# ---------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------

print()
print("EXAMPLE TEXTS")
print("=============")


dataset = load_dataset(
    "shahxeebhassan/human_vs_ai_sentences",
    split="train",
    streaming=True
)


shown = {
    "Human": 0,
    "AI": 0
}


for item in dataset:

    label = item.get(
        "label",
        None
    )


    if label == 0:

        label_name = "Human"

    elif label == 1:

        label_name = "AI"

    else:

        continue


    if shown[label_name] >= 3:
        continue


    text = str(
        item.get(
            "text",
            ""
        )
    ).strip()


    print()
    print(
        f"{label_name}:"
    )

    print(
        text[:500]
    )


    shown[label_name] += 1


    if (
        shown["Human"] >= 3
        and
        shown["AI"] >= 3
    ):

        break


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print()
print(
    "SENTENCE DATASET INSPECTION COMPLETED"
)