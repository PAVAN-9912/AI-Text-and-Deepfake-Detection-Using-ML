import os
import glob
import pandas as pd
import numpy as np

print("="*70)
print("1. DATASET FILES INSPECTION")
print("="*70)
text_datasets = glob.glob("datasets/text/**/*", recursive=True)
for p in text_datasets:
    if os.path.isfile(p):
        size_kb = os.path.getsize(p) / 1024
        print(f"  {p} ({size_kb:.1f} KB)")

print("\n" + "="*70)
print("2. HELD-OUT VALIDATION SET INSPECTION (1,089 SAMPLES)")
print("="*70)
val_path = "models/text_final_v2_improved/validation_results.csv"
if os.path.exists(val_path):
    df_val = pd.read_csv(val_path)
    print(f"Validation File: {val_path}")
    print(f"Total Rows: {len(df_val)}")
    print(f"Columns: {list(df_val.columns)}")
    print("\nClass Distribution (0 = Human, 1 = AI):")
    print(df_val["label"].value_counts())
    print("\nLength Group Distribution:")
    print(df_val["length_group"].value_counts())
    print("\nWord Count Stats:")
    print(df_val["word_count"].describe())

print("\n" + "="*70)
print("3. TRAINING DATASET INSPECTION")
print("="*70)
# Check training files or scripts in src/
train_candidates = [
    "datasets/text/HC3/feature_development.csv",
    "datasets/text/HC3/development.csv",
    "datasets/text/HC3/train.csv",
    "datasets/text/RAID/train.csv"
]
for tc in train_candidates:
    if os.path.exists(tc):
        df_t = pd.read_csv(tc)
        print(f"File: {tc} | Rows: {len(df_t)} | Columns: {list(df_t.columns[:6])}")
        if "label" in df_t.columns:
            print(f"  Class balance: {df_t['label'].value_counts().to_dict()}")

# Inspect text training script
train_scripts = glob.glob("src/train*.py") + glob.glob("src/*v2_improved*.py")
print("\nRelated Training/Evaluation Scripts in src/:")
for ts in train_scripts:
    print(f"  - {ts}")
