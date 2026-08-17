import os
import joblib
import pandas as pd

from scipy.sparse import hstack

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = r"models\text_generalized"

TEST_FILE = (
    r"datasets\text\HC3\external_test.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("EXTERNAL TEST — GENERALIZED TEXT MODEL")
print("=======================================")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "text_classifier.pkl",
    "text_config.pkl"
]


for filename in required_files:

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing model file: {path}"
        )


if not os.path.exists(TEST_FILE):

    raise FileNotFoundError(
        f"External test file not found: {TEST_FILE}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading generalized text model...")


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


config = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_config.pkl"
    )
)


print(
    "Model type:",
    config["model_type"]
)


# ============================================================
# LOAD EXTERNAL DATA
# ============================================================

df = pd.read_csv(
    TEST_FILE
)


if "text" not in df.columns:

    raise ValueError(
        "External dataset must contain 'text'."
    )


if "label" not in df.columns:

    raise ValueError(
        "External dataset must contain 'label'."
    )


df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)


df["label"] = (
    df["label"]
    .astype(int)
)


y_true = (
    df["label"]
    .values
)


print()
print(
    "External samples:",
    len(df)
)


print()
print("LABEL DISTRIBUTION")
print("==================")


print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# WORD FEATURES
# ============================================================

print()
print("BUILDING WORD FEATURES")
print("======================")


X_word = (
    word_vectorizer.transform(
        df["text"]
    )
)


print(
    "Word feature shape:",
    X_word.shape
)


# ============================================================
# CHARACTER FEATURES
# ============================================================

print()
print("BUILDING CHARACTER FEATURES")
print("============================")


X_char = (
    char_vectorizer.transform(
        df["text"]
    )
)


print(
    "Character feature shape:",
    X_char.shape
)


# ============================================================
# COMBINE
# ============================================================

X = hstack(
    [
        X_word,
        X_char
    ]
).tocsr()


print()
print(
    "Combined feature shape:",
    X.shape
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("RUNNING PREDICTIONS")
print("===================")


predictions = (
    classifier.predict(
        X
    )
)


probabilities = (
    classifier.predict_proba(
        X
    )[:, 1]
)


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


# ============================================================
# RESULTS
# ============================================================

print()
print("EXTERNAL GENERALIZATION RESULTS")
print("================================")


print(
    f"Accuracy : {accuracy:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall   : {recall:.4f}"
)


print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("================")


cm = confusion_matrix(
    y_true,
    predictions
)


print(
    cm
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
# SAMPLE ANALYSIS
# ============================================================

print()
print("SAMPLE-BY-SAMPLE ANALYSIS")
print("==========================")


for index, row in df.iterrows():

    actual = (
        "AI"
        if int(row["label"]) == 1
        else "Human"
    )


    predicted = (
        "AI"
        if int(predictions[index]) == 1
        else "Human"
    )


    ai_probability = float(
        probabilities[index]
    )


    confidence = max(
        ai_probability,
        1.0 - ai_probability
    )


    result = (
        "CORRECT"
        if actual == predicted
        else "WRONG"
    )


    print(
        f"{index + 1:02d}. "
        f"Actual={actual:<6} "
        f"Predicted={predicted:<6} "
        f"AI={ai_probability * 100:6.2f}% "
        f"Confidence={confidence * 100:6.2f}% "
        f"{result}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "actual_label":
            y_true,

        "predicted_label":
            predictions,

        "ai_probability":
            probabilities,

        "human_probability":
            1.0 - probabilities,

        "correct":
            predictions == y_true,

        "text":
            df["text"].values
    }
)


results["actual_name"] = (
    results["actual_label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


results["predicted_name"] = (
    results["predicted_label"]
    .map(
        {
            0: "Human",
            1: "AI"
        }
    )
)


OUTPUT_FILE = (
    r"datasets\text\HC3"
    r"\external_generalized_results.csv"
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    "Detailed results saved:"
)

print(
    OUTPUT_FILE
)


print()
print(
    "EXTERNAL GENERALIZATION TEST COMPLETED"
)