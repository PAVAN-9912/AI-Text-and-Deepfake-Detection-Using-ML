from collections import Counter
from datasets import load_dataset

print("RAID LABEL INSPECTION")
print("=====================")

dataset = load_dataset(
    "liamdugan/raid",
    split="extra",
    streaming=True
)

counts = Counter()

limit = 10000

for index, row in enumerate(dataset):

    counts[str(row["model"])] += 1

    if (index + 1) % 250 == 0:
        print(f"Scanned {index + 1}/{limit}")

    if index + 1 >= limit:
        break

print("\nLABEL DISTRIBUTION")
print("==================")

for label, count in counts.items():
    print(f"{label}: {count}")