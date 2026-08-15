import json
import csv

INPUT_FILE = r"datasets\text\HC3\all.jsonl"
OUTPUT_FILE = r"datasets\text\HC3\clean_hc3.csv"

samples = []
seen = set()

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for record_id, line in enumerate(file):
        record = json.loads(line)

        source = record["source"]

        # Human-written answers
        for text in record["human_answers"]:
            text = text.strip()

            if text and text not in seen:
                samples.append({
                    "text": text,
                    "label": 0,
                    "source": source,
                    "record_id": record_id
                })
                seen.add(text)

        # AI-generated answers
        for text in record["chatgpt_answers"]:
            text = text.strip()

            if text and text not in seen:
                samples.append({
                    "text": text,
                    "label": 1,
                    "source": source,
                    "record_id": record_id
                })
                seen.add(text)

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["text", "label", "source", "record_id"]
    )

    writer.writeheader()
    writer.writerows(samples)

print("Dataset preparation completed.")
print("Total unique samples:", len(samples))
print("Output:", OUTPUT_FILE)