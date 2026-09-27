import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# LOAD EXISTING VALIDATION PREDICTIONS
# =========================================================

d = pd.read_csv(
    "models/text_final_v2_improved/validation_results.csv"
).reset_index(drop=True)

y = d["label"].astype(int).values

groups = [
    "very_short",
    "short",
    "medium",
    "long"
]


# =========================================================
# 5-FOLD STRATIFIED TEST
# =========================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_y = []
all_fixed = []
all_length_aware = []

print()
print("5-FOLD THRESHOLD STABILITY TEST")
print("=" * 100)


for fold, (train_idx, test_idx) in enumerate(
    skf.split(d, y),
    start=1
):

    train_df = d.iloc[train_idx]
    test_df = d.iloc[test_idx]

    rules = {}

    # -----------------------------------------------------
    # Find threshold independently for each length group
    # using ONLY the training portion of this fold.
    # -----------------------------------------------------

    for group in groups:

        group_train = train_df[
            train_df["length_group"] == group
        ]

        best_f1 = -1
        best_threshold = 0.41

        for threshold in np.arange(
            0.20,
            0.51,
            0.01
        ):

            predictions = (
                group_train["probability"] >= threshold
            ).astype(int)

            score = f1_score(
                group_train["label"],
                predictions,
                zero_division=0
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        rules[group] = best_threshold

    # -----------------------------------------------------
    # Evaluate on completely held-out fold
    # -----------------------------------------------------

    fixed_predictions = (
        test_df["probability"] >= 0.41
    ).astype(int)

    length_predictions = (
        test_df["probability"]
        >= test_df["length_group"].map(rules)
    ).astype(int)

    all_y.extend(
        test_df["label"].astype(int)
    )

    all_fixed.extend(
        fixed_predictions
    )

    all_length_aware.extend(
        length_predictions
    )

    print(
        f"Fold {fold}: "
        f"very_short={rules['very_short']:.2f}, "
        f"short={rules['short']:.2f}, "
        f"medium={rules['medium']:.2f}, "
        f"long={rules['long']:.2f}"
    )


# =========================================================
# AGGREGATED OUT-OF-FOLD RESULTS
# =========================================================

print()
print("AGGREGATED OUT-OF-FOLD RESULT")
print("=" * 100)


for name, predictions in [
    ("Fixed 0.41", all_fixed),
    ("Length-aware", all_length_aware)
]:

    accuracy = accuracy_score(
        all_y,
        predictions
    )

    precision = precision_score(
        all_y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_y,
        predictions,
        zero_division=0
    )

    fp = sum(
        (actual == 0 and pred == 1)
        for actual, pred in zip(
            all_y,
            predictions
        )
    )

    fn = sum(
        (actual == 1 and pred == 0)
        for actual, pred in zip(
            all_y,
            predictions
        )
    )

    print(
        f"{name:15s} | "
        f"Acc={accuracy * 100:.2f}% | "
        f"Prec={precision * 100:.2f}% | "
        f"Rec={recall * 100:.2f}% | "
        f"F1={f1 * 100:.2f}% | "
        f"FP={fp} | "
        f"FN={fn}"
    )