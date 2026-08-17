import os
import joblib
import pandas as pd

from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = r"models\text_generalized"

DATA_FILE = r"datasets\text\RAID\raid_external.csv"


# ============================================================
# LOAD DATA
# ============================================================

print()
print("RAID EXTERNAL TEST — TF-IDF MODEL")
print("=================================")

df = pd.read_csv(DATA_FILE)

print()
print("External samples:", len(df))

print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label_name"].value_counts()
)


print()
print("RAID MODEL DISTRIBUTION")
print("=======================")

print(
    df["model"].value_counts()
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("LOADING SAVED MODEL")
print("===================")

word_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)

char_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)

classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_classifier.pkl"
    )
)


# ============================================================
# BUILD FEATURES
# ============================================================

print()
print("BUILDING WORD FEATURES")
print("======================")

X_word = word_vectorizer.transform(
    df["text"].fillna("")
)

print(
    "Word feature shape:",
    X_word.shape
)


print()
print("BUILDING CHARACTER FEATURES")
print("============================")

X_char = char_vectorizer.transform(
    df["text"].fillna("")
)

print(
    "Character feature shape:",
    X_char.shape
)


X = hstack(
    [
        X_word,
        X_char
    ]
).tocsr()


print()
print("Combined feature shape:", X.shape)


# ============================================================
# PREDICTION
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")

y_true = df["label"].astype(int).values

predictions = classifier.predict(
    X
)

probabilities = classifier.predict_proba(
    X
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    probabilities
)


# ============================================================
# RESULTS
# ============================================================

print()
print("RAID TF-IDF RESULTS")
print("===================")

print(
    "Accuracy :", round(accuracy, 4)
)

print(
    "Precision:", round(precision, 4)
)

print(
    "Recall   :", round(recall, 4)
)

print(
    "F1 Score :", round(f1, 4)
)

print(
    "ROC-AUC  :", round(auc, 4)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")

print(
    confusion_matrix(
        y_true,
        predictions
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("CLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_true,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# PERFORMANCE BY AI GENERATOR
# ============================================================

print()
print("PERFORMANCE BY RAID MODEL")
print("=========================")

for model_name in sorted(
    df["model"].unique()
):

    subset = (
        df["model"] == model_name
    )

    subset_true = y_true[subset]
    subset_pred = predictions[subset]

    subset_accuracy = accuracy_score(
        subset_true,
        subset_pred
    )

    print(
        f"{model_name:12s} "
        f"Accuracy={subset_accuracy:.4f} "
        f"Samples={subset.sum()}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = df.copy()

results["prediction"] = predictions

results["ai_probability"] = probabilities

results["prediction_name"] = (
    results["prediction"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)

output_file = (
    r"datasets\text\RAID"
    r"\raid_tfidf_results.csv"
)

results.to_csv(
    output_file,
    index=False
)


print()
print("Detailed results saved:")
print(output_file)

print()
print("RAID TF-IDF EVALUATION COMPLETED")