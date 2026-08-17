import json
from collections import Counter


PATH = r"datasets\text\HC3\all.jsonl"

MAX_SAMPLES = 10000


print()
print("RAW HC3 DATA INSPECTION")
print("=======================")

print()
print("File:", PATH)
print("Scanning first", MAX_SAMPLES, "records...")


label_counts = Counter()
length_counts = Counter()
length_label_counts = Counter()

source_counts = Counter()

first_record = None


def length_group(word_count):

    if word_count <= 30:
        return "short"

    elif word_count <= 100:
        return "medium"

    else:
        return "long"


with open(
    PATH,
    "r",
    encoding="utf-8"
) as f:

    for index, line in enumerate(f):

        if index >= MAX_SAMPLES:
            break

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)

        except json.JSONDecodeError:
            continue


        if first_record is None:

            first_record = record


        # -------------------------------------------------
        # Show possible source/category
        # -------------------------------------------------

        for key in [
            "source",
            "category",
            "domain"
        ]:

            if key in record:

                source_counts[
                    str(record[key])
                ] += 1

                break


        # -------------------------------------------------
        # HC3 RAW FORMAT
        #
        # Usually contains:
        # human_answers
        # chatgpt_answers
        # -------------------------------------------------

        human_answers = record.get(
            "human_answers",
            []
        )

        ai_answers = record.get(
            "chatgpt_answers",
            []
        )


        # -------------------------------------------------
        # HUMAN
        # -------------------------------------------------

        for text in human_answers:

            text = str(text)

            words = text.split()

            count = len(words)

            group = length_group(
                count
            )

            label_counts["Human"] += 1

            length_counts[group] += 1

            length_label_counts[
                (group, "Human")
            ] += 1


        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        for text in ai_answers:

            text = str(text)

            words = text.split()

            count = len(words)

            group = length_group(
                count
            )

            label_counts["AI"] += 1

            length_counts[group] += 1

            length_label_counts[
                (group, "AI")
            ] += 1


        if (index + 1) % 1000 == 0:

            print(
                f"Processed {index + 1} records..."
            )


# =========================================================
# FIRST RECORD
# =========================================================

print()
print("RAW RECORD STRUCTURE")
print("====================")

if first_record:

    print(
        "Keys:",
        list(first_record.keys())
    )

    for key, value in first_record.items():

        if isinstance(value, list):

            print(
                f"{key}: list "
                f"({len(value)} items)"
            )

            if value:

                print(
                    "  Sample:",
                    str(value[0])[:300]
                )

        else:

            print(
                f"{key}:",
                str(value)[:300]
            )


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print()
print("LABEL DISTRIBUTION")
print("==================")

for label in [
    "Human",
    "AI"
]:

    print(
        f"{label:10s}: "
        f"{label_counts[label]}"
    )


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print()
print("LENGTH × LABEL DISTRIBUTION")
print("===========================")

print(
    f"{'Length':10s}"
    f"{'Human':10s}"
    f"{'AI':10s}"
)


for group in [
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
        f"{group:10s}"
        f"{human:<10d}"
        f"{ai:<10d}"
    )


# =========================================================
# SOURCE DISTRIBUTION
# =========================================================

print()
print("SOURCE / CATEGORY DISTRIBUTION")
print("==============================")

for key, value in source_counts.most_common():

    print(
        f"{key:30s}: {value}"
    )


# =========================================================
# FINAL
# =========================================================

print()
print(
    "RAW HC3 INSPECTION COMPLETED"
)