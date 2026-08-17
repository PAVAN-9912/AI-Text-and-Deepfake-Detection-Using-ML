from datasets import load_dataset
from collections import Counter


print()
print("AITEXTBENCH-V2 INSPECTION")
print("=========================")

print()
print("Loading dataset in streaming mode...")
print("The full dataset will NOT be downloaded.")

ds = load_dataset(
    "btufts/AITextBench-V2",
    split="train",
    streaming=True
)


# ---------------------------------------------------------
# INSPECT FIRST SAMPLE
# ---------------------------------------------------------

print()
print("READING FIRST SAMPLE")
print("====================")

first = next(iter(ds))

print()
print("COLUMNS")
print("=======")

for key in first.keys():
    value = first[key]

    print(
        f"{key:15s}: "
        f"{str(value)[:300]}"
    )


# ---------------------------------------------------------
# SCAN DATA
# ---------------------------------------------------------

print()
print("SCANNING FIRST 10000 SAMPLES")
print("============================")

ds = load_dataset(
    "btufts/AITextBench-V2",
    split="train",
    streaming=True
)


MAX_SAMPLES = 10000

label_counter = Counter()
prompt_counter = Counter()
task_counter = Counter()

length_counter = Counter()

length_by_label = Counter()


for i, item in enumerate(ds):

    if i >= MAX_SAMPLES:
        break


    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    text = str(
        item.get(
            "text",
            ""
        )
    )


    words = text.split()

    word_count = len(words)


    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    prompt = str(
        item.get(
            "prompt",
            ""
        )
    )


    task = str(
        item.get(
            "task",
            ""
        )
    )


    # -----------------------------------------------------
    # LABEL INFERENCE
    #
    # AITextBench uses prompt:
    # human = human response
    # base/template/rewrite = AI generation
    # -----------------------------------------------------

    if prompt == "human":

        label = "Human"

    elif prompt in [
        "base",
        "template",
        "rewrite"
    ]:

        label = "AI"

    else:

        label = "Unknown"


    # -----------------------------------------------------
    # LENGTH
    # -----------------------------------------------------

    if word_count <= 30:

        length_group = "short"

    elif word_count <= 100:

        length_group = "medium"

    else:

        length_group = "long"


    # -----------------------------------------------------
    # COUNTERS
    # -----------------------------------------------------

    label_counter[label] += 1

    prompt_counter[prompt] += 1

    task_counter[task] += 1

    length_counter[length_group] += 1

    length_by_label[
        (
            length_group,
            label
        )
    ] += 1


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if (i + 1) % 1000 == 0:

        print(
            f"Scanned {i + 1}/{MAX_SAMPLES}"
        )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("LABEL DISTRIBUTION")
print("==================")

for key, value in label_counter.items():

    print(
        f"{key:10s}: {value}"
    )


print()
print("PROMPT DISTRIBUTION")
print("===================")

for key, value in prompt_counter.items():

    print(
        f"{key:10s}: {value}"
    )


print()
print("TASK DISTRIBUTION")
print("=================")

for key, value in task_counter.most_common():

    print(
        f"{key:30s}: {value}"
    )


print()
print("LENGTH DISTRIBUTION")
print("===================")

for key in [
    "short",
    "medium",
    "long"
]:

    print(
        f"{key:10s}: "
        f"{length_counter[key]}"
    )


print()
print("LENGTH × LABEL DISTRIBUTION")
print("===========================")

print(
    f"{'Length':10s}"
    f"{'Human':10s}"
    f"{'AI':10s}"
    f"{'Unknown':10s}"
)


for length in [
    "short",
    "medium",
    "long"
]:

    human = length_by_label[
        (length, "Human")
    ]

    ai = length_by_label[
        (length, "AI")
    ]

    unknown = length_by_label[
        (length, "Unknown")
    ]


    print(
        f"{length:10s}"
        f"{human:<10d}"
        f"{ai:<10d}"
        f"{unknown:<10d}"
    )


print()
print(
    "AITEXTBENCH INSPECTION COMPLETED"
)