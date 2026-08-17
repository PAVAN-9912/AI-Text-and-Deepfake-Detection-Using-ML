from collections import Counter

from datasets import load_dataset


print()
print("RAID EXTRA DOMAIN INSPECTION")
print("============================")


dataset = load_dataset(
    "liamdugan/raid",
    name="raid",
    split="extra",
    streaming=True
)


model_counts = Counter()
domain_counts = Counter()
model_domain_counts = Counter()


MAX_SAMPLES = 10000


print()
print(f"Scanning first {MAX_SAMPLES} EXTRA samples...")
print()


for i, item in enumerate(dataset, start=1):

    model = str(
        item.get("model", "unknown")
    )

    domain = str(
        item.get("domain", "unknown")
    )

    model_counts[model] += 1
    domain_counts[domain] += 1
    model_domain_counts[(model, domain)] += 1

    if i % 1000 == 0:
        print(f"Scanned {i}/{MAX_SAMPLES}")

    if i >= MAX_SAMPLES:
        break


print()
print("MODEL DISTRIBUTION")
print("==================")

for model, count in model_counts.most_common():
    print(f"{model:15s}: {count}")


print()
print("DOMAIN DISTRIBUTION")
print("===================")

for domain, count in domain_counts.most_common():
    print(f"{domain:15s}: {count}")


print()
print("MODEL × DOMAIN DISTRIBUTION")
print("============================")

for (model, domain), count in sorted(
    model_domain_counts.items(),
    key=lambda x: (x[0][1], x[0][0])
):

    print(
        f"{model:15s} "
        f"{domain:15s} "
        f"{count}"
    )


print()
print("INSPECTION COMPLETED")