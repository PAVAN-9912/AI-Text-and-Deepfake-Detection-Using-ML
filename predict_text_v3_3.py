import os
import re
import sys
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

# Paths
MODEL_DIR = r"models\text_final_v3_3"

# Load saved artifacts
with open(os.path.join(MODEL_DIR, "word_tfidf_vectorizer.pkl"), "rb") as f:
    word_vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "char_tfidf_vectorizer.pkl"), "rb") as f:
    char_vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "linguistic_scaler.pkl"), "rb") as f:
    linguistic_scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "text_classifier.pkl"), "rb") as f:
    classifier = pickle.load(f)

with open(os.path.join(MODEL_DIR, "text_config.pkl"), "rb") as f:
    config = pickle.load(f)

transformer_name = config.get("transformer", "sentence-transformers/all-MiniLM-L6-v2")
transformer_model = SentenceTransformer(transformer_name)

length_thresholds = config["length_thresholds"]
alpha_short_decay = config.get("short_interaction_alpha", 0.75)

# Regex patterns
contractions = [
    r"can't", r"won't", r"don't", r"doesn't", r"didn't", r"isn't", r"aren't",
    r"wasn't", r"weren't", r"haven't", r"hasn't", r"hadn't", r"i'm", r"i've",
    r"i'll", r"i'd", r"you're", r"you've", r"you'll", r"you'd", r"he's",
    r"she's", r"it's", r"that's", r"we're", r"we've", r"we'll", r"they'd",
    r"they're", r"they've", r"they'll", r"couldn't", r"shouldn't", r"wouldn't",
    r"there's", r"what's", r"let's", r"who's"
]
contraction_pattern = r"\b(" + "|".join(c.replace("'", r"['\u2019]") for c in contractions) + r")\b"
contraction_regex = re.compile(contraction_pattern, re.IGNORECASE)

pronouns = [
    r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b",
    r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b",
    r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b", r"\byourselves\b"
]
pronoun_regex = re.compile("|".join(pronouns), re.IGNORECASE)

code_kw_regex = re.compile(
    r"(\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:\(]|\breturn\b|\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import|\bwhile\s+.*:|\bfor\s+\w+\s+in\s+|\belif\s+.*:|\bexcept\s*.*:|\blambda\s+.*:|\bprint\s*\(|\bself\.\w+)",
    re.IGNORECASE
)
code_sym_regex = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\b\d+\s*//\s*\d+\b|->|::|\{|\}|\[\s*\]|\(\s*\)|;\s*$)",
    re.MULTILINE
)

def get_length_group(text):
    wc = len(str(text).split())
    if wc <= 10:
        return "very_short"
    elif wc <= 30:
        return "short"
    elif wc <= 60:
        return "medium"
    else:
        return "long"

def extract_features_single(text):
    text = str(text).strip()
    words = text.split()
    word_count = len(words)
    wc_safe = max(word_count, 1)

    # 1. Base linguistic features
    sentence_count = sum(1 for char in text if char in ".!?")
    if sentence_count == 0:
        sentence_count = 1

    avg_sentence_length = word_count / max(sentence_count, 1)

    if word_count > 0:
        unique_words = len(set(w.lower() for w in words))
        vocabulary_diversity = unique_words / word_count
    else:
        vocabulary_diversity = 0.0

    log_word_count = np.log1p(word_count)

    if word_count <= 10:
        length_bucket = 0.0
    elif word_count <= 30:
        length_bucket = 1.0
    elif word_count <= 60:
        length_bucket = 2.0
    else:
        length_bucket = 3.0

    # 2. Style features
    contraction_count = len(contraction_regex.findall(text))
    contraction_density = contraction_count / wc_safe

    pronoun_count = len(pronoun_regex.findall(text))
    pronoun_density = pronoun_count / wc_safe

    question_mark_density = text.count('?') / wc_safe
    exclamation_density = len(re.findall(r'!(?!=)', text)) / wc_safe
    ellipsis_density = len(re.findall(r'\.{2,}', text)) / wc_safe

    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    uppercase_word_ratio = upper_words / wc_safe

    irreg_caps = sum(1 for w in words if any(c.isupper() for c in w[1:]) and not w.isupper())
    irregular_caps_ratio = irreg_caps / wc_safe

    # 3. Code / Syntax features
    code_kw_count = len(code_kw_regex.findall(text))
    code_keyword_density = code_kw_count / wc_safe

    code_sym_count = len(code_sym_regex.findall(text))
    code_symbol_density = code_sym_count / wc_safe

    function_def_flag = 1.0 if bool(re.search(r'\bdef\s+[a-zA-Z_]\w*\s*\(', text)) else 0.0
    import_stmt_flag = 1.0 if bool(re.search(r'(\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import)', text)) else 0.0
    has_indentation = 1.0 if bool(re.search(r'^\s{2,}\S', text, re.MULTILINE)) else 0.0

    code_likelihood = min(1.0, (code_keyword_density * 2.5 + code_symbol_density * 2.0 + function_def_flag * 0.4 + import_stmt_flag * 0.3 + has_indentation * 0.2))

    raw_style = np.array([[
        log_word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity,
        length_bucket,
        contraction_density,
        pronoun_density,
        question_mark_density,
        exclamation_density,
        ellipsis_density,
        uppercase_word_ratio,
        irregular_caps_ratio,
        code_keyword_density,
        code_symbol_density,
        function_def_flag,
        import_stmt_flag,
        code_likelihood
    ]], dtype=np.float32)

    return raw_style, word_count, code_likelihood

def predict_text(text):
    text = str(text).strip()
    if not text:
        return {
            "prediction": "HUMAN",
            "probability": 0.0,
            "length_group": "very_short",
            "threshold": length_thresholds.get("very_short", 0.5),
            "code_likelihood": 0.0
        }

    lg = get_length_group(text)
    threshold = length_thresholds.get(lg, 0.50)

    # Transform TF-IDF
    X_word = word_vectorizer.transform([text])
    X_char = char_vectorizer.transform([text])

    # Embedding
    emb = transformer_model.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    
    # Style & Code
    raw_style, word_count, code_likelihood = extract_features_single(text)
    X_style_scaled = linguistic_scaler.transform(raw_style)

    # Short interaction
    decay = np.array([[alpha_short_decay * np.exp(-word_count / 15.0)]], dtype=np.float32)
    X_short_inter = csr_matrix((emb * decay).astype(np.float32))

    # Feature matrix
    X_full = hstack([
        X_word,
        X_char,
        csr_matrix(emb),
        X_short_inter,
        csr_matrix(X_style_scaled)
    ], format="csr")

    prob = float(classifier.predict_proba(X_full)[:, 1][0])
    is_ai = prob >= threshold
    verdict = "AI" if is_ai else "HUMAN"

    return {
        "prediction": verdict,
        "probability": round(prob, 4),
        "length_group": lg,
        "threshold": round(threshold, 2),
        "code_likelihood": round(code_likelihood, 4)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "def count_digit(n):\n    while n != 0:\n        count += 1\n        n //= 10\n    return count"

    result = predict_text(input_text)
    print("=" * 60)
    print("V3.3 AI TEXT DETECTOR PREDICTION")
    print("=" * 60)
    print(f"Input Text: {repr(input_text[:80])}...")
    print(f"Prediction     : {result['prediction']}")
    print(f"Probability    : {result['probability']:.4f}")
    print(f"Length Group   : {result['length_group']}")
    print(f"Threshold      : {result['threshold']:.2f}")
    print(f"Code Likelihood: {result['code_likelihood']:.4f}")
    print("=" * 60)
