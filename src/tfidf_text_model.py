import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


TRAIN_FILE = r"datasets\text\HC3\train.csv"
VALIDATION_FILE = r"datasets\text\HC3\validation.csv"
TEST_FILE = r"datasets\text\HC3\test.csv"


# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train_text = train_df["text"].fillna("")
X_validation_text = validation_df["text"].fillna("")
X_test_text = test_df["text"].fillna("")

y_train = train_df["label"]
y_validation = validation_df["label"]
y_test = test_df["label"]


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)


# IMPORTANT:
# Fit only on training data
X_train = vectorizer.fit_transform(X_train_text)

# Transform validation and test using the same vectorizer
X_validation = vectorizer.transform(X_validation_text)
X_test = vectorizer.transform(X_test_text)


print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))
print("Training matrix shape:", X_train.shape)


# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# Validation
validation_pred = model.predict(X_validation)

print("\nVALIDATION RESULTS")
print("------------------")
print("Accuracy :", round(accuracy_score(y_validation, validation_pred), 4))
print("Precision:", round(precision_score(y_validation, validation_pred), 4))
print("Recall   :", round(recall_score(y_validation, validation_pred), 4))
print("F1 Score :", round(f1_score(y_validation, validation_pred), 4))


# Test
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