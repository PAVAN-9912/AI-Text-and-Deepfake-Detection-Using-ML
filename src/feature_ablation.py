import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = r"datasets\text\HC3\research_features.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("FEATURE ABLATION STUDY")
print("=====================")

print("Total samples:", len(df))


# --------------------------------------------------
# Create transformed features
# --------------------------------------------------

df["burstiness_missing"] = (
    df["burstiness"].isna().astype(int)
)

df["log_perplexity"] = np.log1p(
    df["perplexity"]
)

df["log_burstiness"] = np.log1p(
    df["burstiness"]
)


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

statistics = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity"
]

statistics_perplexity = statistics + [
    "log_perplexity"
]

statistics_burstiness = statistics + [
    "log_burstiness",
    "burstiness_missing"
]

all_features = statistics + [
    "log_perplexity",
    "log_burstiness",
    "burstiness_missing"
]


experiments = {
    "A: Statistics only":
        statistics,

    "B: Statistics + Perplexity":
        statistics_perplexity,

    "C: Statistics + Burstiness":
        statistics_burstiness,

    "D: Statistics + Perplexity + Burstiness":
        all_features
}


y = df["label"]


# --------------------------------------------------
# Cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


for name, features in experiments.items():

    print("\n" + name)
    print("-" * len(name))

    X = df[features]

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []


    for train_index, val_index in cv.split(X, y):

        X_train = X.iloc[train_index]
        X_val = X.iloc[val_index]

        y_train = y.iloc[train_index]
        y_val = y.iloc[val_index]


        pipeline = Pipeline([

            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ])


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = pipeline.predict(
            X_val
        )


        accuracies.append(
            accuracy_score(
                y_val,
                predictions
            )
        )

        precisions.append(
            precision_score(
                y_val,
                predictions
            )
        )

        recalls.append(
            recall_score(
                y_val,
                predictions
            )
        )

        f1_scores.append(
            f1_score(
                y_val,
                predictions
            )
        )


    avg_accuracy = np.mean(
        accuracies
    )

    avg_precision = np.mean(
        precisions
    )

    avg_recall = np.mean(
        recalls
    )

    avg_f1 = np.mean(
        f1_scores
    )


    print(
        "Accuracy :",
        round(avg_accuracy, 4)
    )

    print(
        "Precision:",
        round(avg_precision, 4)
    )

    print(
        "Recall   :",
        round(avg_recall, 4)
    )

    print(
        "F1 Score :",
        round(avg_f1, 4)
    )


    results.append({

        "Model": name,

        "Accuracy":
            avg_accuracy,

        "Precision":
            avg_precision,

        "Recall":
            avg_recall,

        "F1 Score":
            avg_f1
    })


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n\nFINAL ABLATION COMPARISON")
print("=========================")

print(
    results_df.round(4).to_string(
        index=False
    )
)