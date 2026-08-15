import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


TRAIN_FILE = r"datasets\text\HC3\train.csv"
VALIDATION_FILE = r"datasets\text\HC3\validation.csv"
TEST_FILE = r"datasets\text\HC3\test.csv"


def extract_features(df):
    text = df["text"].fillna("")

    features = pd.DataFrame()

    features["word_count"] = text.str.findall(r"\b\w+\b").str.len()

    features["sentence_count"] = text.str.count(r"[.!?]").clip(lower=1)

    features["avg_sentence_length"] = (
        features["word_count"] / features["sentence_count"]
    )

    unique_words = (
        text.str.lower()
        .str.findall(r"\b\w+\b")
        .apply(lambda x: len(set(x)))
    )

    features["vocabulary_diversity"] = (
        unique_words / features["word_count"].replace(0, 1)
    )

    return features


# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)


# Extract features
X_train = extract_features(train_df)
X_validation = extract_features(validation_df)
X_test = extract_features(test_df)

y_train = train_df["label"]
y_validation = validation_df["label"]
y_test = test_df["label"]


# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)


# Train baseline model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# Validation prediction
validation_pred = model.predict(X_validation)

print("\nVALIDATION RESULTS")
print("------------------")
print("Accuracy :", round(accuracy_score(y_validation, validation_pred), 4))
print("Precision:", round(precision_score(y_validation, validation_pred), 4))
print("Recall   :", round(recall_score(y_validation, validation_pred), 4))
print("F1 Score :", round(f1_score(y_validation, validation_pred), 4))


# Final test prediction
test_pred = model.predict(X_test)

print("\nTEST RESULTS")
print("------------")
print("Accuracy :", round(accuracy_score(y_test, test_pred), 4))
print("Precision:", round(precision_score(y_test, test_pred), 4))
print("Recall   :", round(recall_score(y_test, test_pred), 4))
print("F1 Score :", round(f1_score(y_test, test_pred), 4))


# Confusion matrix
print("\nCONFUSION MATRIX")
print("----------------")
print(confusion_matrix(y_test, test_pred))