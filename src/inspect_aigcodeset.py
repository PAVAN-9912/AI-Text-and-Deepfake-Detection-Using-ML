from datasets import load_dataset
from collections import Counter


print()
print("AIGCODESET INSPECTION")
print("=====================")
print()

print("Loading AIGCodeSet in streaming mode...")
print("The full dataset will NOT be downloaded.")
print()

ds = load_dataset(
    "basakdemirok/AIGCodeSet",
    split="train",
    streaming=True
)

MAX_SCAN = 5000

label_counts = Counter()
language_counts = Counter()
llm_counts = Counter()

examples = []

scanned = 0

for item in ds:

    scanned += 1

    if scanned % 500 == 0:
        print(
            f"Scanned {scanned}/{MAX_SCAN}"
        )

    label = str(
        item.get("label", "")
    ).strip()

    llm = str(
        item.get("LLM", "")
    ).strip()

    code = str(
        item.get("code", "")
    ).strip()

    label_counts[label] += 1

    llm_counts[llm] += 1

    if len(code) > 0:
        examples.append(
            (
                label,
                llm,
                code
            )
        )

    if scanned >= MAX_SCAN:
        break


print()
print("SCAN COMPLETED")
print("===============")

print(
    f"Samples scanned: {scanned}"
)


print()
print("LABEL DISTRIBUTION")
print("==================")

for key, value in label_counts.items():

    print(
        f"{key:20s}: {value}"
    )


print()
print("GENERATOR DISTRIBUTION")
print("======================")

for key, value in llm_counts.items():

    print(
        f"{key:20s}: {value}"
    )


print()
print("EXAMPLE CODE")
print("============")

for i, example in enumerate(
    examples[:6],
    start=1
):

    label, llm, code = example

    print()
    print(
        f"Example {i}"
    )

    print(
        f"Label     : {label}"
    )

    print(
        f"Generator : {llm}"
    )

    print(
        "Code:"
    )

    print(
        code[:1000]
    )

    print(
        "-" * 70
    )


print()
print("AIGCODESET INSPECTION COMPLETED")