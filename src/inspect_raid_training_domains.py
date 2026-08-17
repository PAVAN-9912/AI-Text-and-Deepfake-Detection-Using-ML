from collections import Counter

from datasets import load_dataset


print()
print("RAID TRAINING DOMAIN INSPECTION")
print("================================")


dataset = load_dataset(
    "liamdugan/raid",
    name="raid",
    split="train",
    streaming=True
)


models = Counter()
domains = Counter()
model_domains = Counter()


# Scan farther into the dataset.
# We are NOT downloading the full dataset.
MAX_SAMPLES = 100000


print()
print(
    f"Scanning first {MAX_SAMPLES} training samples..."
)
print()


for i, item in enumerate(dataset, start=1):

    model = str(
        item.get(
            "model",
            "unknown"
        )
    )

    domain = str(
        item.get(
            "domain",
            "unknown"
        )
    )


    models[model] += 1
    domains[domain] += 1
    model_domains[(model, domain)] += 1


    if i % 5000 == 0:

        print(
            f"Scanned {i}/{MAX_SAMPLES}"
        )


    if i >= MAX_SAMPLES:

        break


print()
print("MODEL DISTRIBUTION")
print("==================")

for model, count in models.most_common():

    print(
        f"{model:15s}: {count}"
    )


print()
print("DOMAIN DISTRIBUTION")
print("===================")

for domain, count in domains.most_common():

    print(
        f"{domain:15s}: {count}"
    )


print()
print("MODEL × DOMAIN DISTRIBUTION")
print("============================")

for (model, domain), count in sorted(
    model_domains.items(),
    key=lambda x: (x[0][1], x[0][0])
):

    print(
        f"{model:15s} "
        f"{domain:15s} "
        f"{count}"
    )


print()
print("INSPECTION COMPLETED")