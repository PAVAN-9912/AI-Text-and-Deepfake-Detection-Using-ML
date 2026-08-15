import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT_FILE = r"datasets\text\HC3\clean_hc3.csv"

TRAIN_FILE = r"datasets\text\HC3\train.csv"
VALIDATION_FILE = r"datasets\text\HC3\validation.csv"
TEST_FILE = r"datasets\text\HC3\test.csv"


# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("Total samples:", len(df))
print("Unique record IDs:", df["record_id"].nunique())


# --------------------------------------------------
# STEP 1: 70% Training + 30% Temporary
# --------------------------------------------------

splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    splitter_1.split(
        df,
        groups=df["record_id"]
    )
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()


# --------------------------------------------------
# STEP 2: Split remaining 30% into
#         15% Validation + 15% Test
# --------------------------------------------------

splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

validation_idx, test_idx = next(
    splitter_2.split(
        temp_df,
        groups=temp_df["record_id"]
    )
)

validation_df = temp_df.iloc[validation_idx].copy()
test_df = temp_df.iloc[test_idx].copy()


# Save datasets
train_df.to_csv(TRAIN_FILE, index=False)
validation_df.to_csv(VALIDATION_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nSplit completed.")

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(validation_df))
print("Test samples:", len(test_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(validation_df["label"].value_counts())

print("\nTest labels:")
print(test_df["label"].value_counts())

print("\nFiles created:")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)