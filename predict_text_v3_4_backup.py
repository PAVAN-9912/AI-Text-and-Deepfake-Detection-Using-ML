import os
import re
import sys
import pickle
import math
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

# Paths
MODEL_DIR = r"models\text_final_v3_4"

# Load saved artifacts
with open(os.path.join(MODEL_DIR, "word_tfidf_vectorizer.pkl"), "rb") as f:
    word_vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "char_tfidf_vectorizer.pkl"), "rb") as f:
    char_vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "multi_char_tfidf_vectorizer.pkl"), "rb") as f:
    multi_char_vectorizer = pickle.load(f)

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
    r"can't", r"cannot", r"won't", r"don't", r"doesn't", r"didn't", r"isn't", r"aren't",
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
    r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b", r"\byourselves\b",
    r"\bthey\b", r"\bthem\b", r"\btheir\b", r"\btheirs\b", r"\bthemselves\b"
]
pronoun_regex = re.compile("|".join(pronouns), re.IGNORECASE)

transition_words = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b", r"\bin addition\b",
    r"\bhowever\b", r"\btherefore\b", r"\bthus\b", r"\bconsequently\b",
    r"\bfirstly\b", r"\bsecondly\b", r"\bfinally\b", r"\bto summarize\b",
    r"\bon the other hand\b", r"\bas a result\b", r"\bin contrast\b"
]
transition_regex = re.compile("|".join(transition_words), re.IGNORECASE)

code_kw_regex = re.compile(
    r"(\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:\(]|\breturn\b|\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import|\bwhile\s+.*:|\bfor\s+\w+\s+in\s+|\belif\s+.*:|\belse\s*:|\btry\s*:|\bexcept\s*.*:|\blambda\b|\byield\b|\bprint\s*\(|\bself\.\w+)",
    re.IGNORECASE
)
code_sym_regex = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\*\*|\b\d+\s*//\s*\d+\b|->|::|\{|\}|\[\s*\]|\(\s*\)|;\s*$)",
    re.MULTILINE
)

german_chars_regex = re.compile(r"[äöüÄÖÜß]")
accented_chars_regex = re.compile(r"[éèêëáàâäãåíìîïóòôöõúùûüñç]", re.IGNORECASE)

def calculate_token_length_entropy(words):
    if not words:
        return 0.0
    lengths = [len(w) for w in words]
    counts = {}
    for l in lengths:
        counts[l] = counts.get(l, 0) + 1
    total = len(lengths)
    entropy = 0.0
    for l, c in counts.items():
        p = c / total
        entropy -= p * math.log2(p)
    return float(entropy)

def calculate_repetitive_ngram_ratio(words, n=3):
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    repeated = total - unique
    return float(repeated / max(total, 1))

def calculate_sentence_length_variability(text, words):
    raw_sentences = re.split(r"[.!?]+", text)
    valid_sentences = [s.strip().split() for s in raw_sentences if s.strip()]
    if len(valid_sentences) <= 1:
        return 0.0
    lens = [len(s) for s in valid_sentences]
    mean_len = np.mean(lens)
    std_len = np.std(lens)
    return float(std_len / max(mean_len, 1.0))

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
    char_count = len(text)
    wc_safe = max(word_count, 1)
    cc_safe = max(char_count, 1)

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

    # 2. Style / Lexical features
    token_entropy = calculate_token_length_entropy(words)
    rep_ngram_ratio = calculate_repetitive_ngram_ratio(words, n=3)
    transition_ratio = len(transition_regex.findall(text)) / wc_safe
    contraction_density = len(contraction_regex.findall(text)) / wc_safe
    pronoun_density = len(pronoun_regex.findall(text)) / wc_safe
    question_density = text.count("?") / wc_safe
    exclamation_density = len(re.findall(r"!(?!=)", text)) / wc_safe
    ellipsis_density = len(re.findall(r"\.{2,}", text)) / wc_safe
    sent_len_var = calculate_sentence_length_variability(text, words)

    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    uppercase_word_ratio = upper_words / wc_safe

    irreg_caps = sum(1 for w in words if any(c.isupper() for c in w[1:]) and not w.isupper())
    irregular_caps_ratio = irreg_caps / wc_safe

    # 3. Code / Syntax features
    code_kw_count = len(code_kw_regex.findall(text))
    code_keyword_density = code_kw_count / wc_safe

    code_sym_count = len(code_sym_regex.findall(text))
    code_symbol_density = code_sym_count / wc_safe

    function_def_flag = 1.0 if bool(re.search(r"\bdef\s+[a-zA-Z_]\w*\s*\(", text)) else 0.0
    import_stmt_flag = 1.0 if bool(re.search(r"(\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import)", text)) else 0.0
    has_indentation = 1.0 if bool(re.search(r"^\s{2,}\S", text, re.MULTILINE)) else 0.0

    code_lines = sum(1 for l in text.splitlines() if re.search(r"(def\s+|class\s+|return|import|^\s{2,}\S|[{};])", l))
    code_line_ratio = code_lines / max(len(text.splitlines()), 1)

    code_likelihood = min(1.0, (code_keyword_density * 2.5 + code_symbol_density * 2.0 + function_def_flag * 0.4 + import_stmt_flag * 0.3 + has_indentation * 0.2 + code_line_ratio * 0.3))

    # 4. Multilingual / Non-English features
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    non_ascii_ratio = non_ascii_count / cc_safe

    german_char_count = len(german_chars_regex.findall(text))
    german_char_density = german_char_count / cc_safe

    accented_char_count = len(accented_chars_regex.findall(text))
    accented_char_density = accented_char_count / cc_safe

    is_german_indicator = 1.0 if (german_char_count > 0 or bool(re.search(r"\b(der|die|das|und|ist|nicht|f[üu]r|ein|eine|einen|einer|vom|von|mit|sich|dem|den)\b", text, re.IGNORECASE))) else 0.0

    raw_scalar = np.array([[
        log_word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity,
        length_bucket,
        token_entropy,
        rep_ngram_ratio,
        transition_ratio,
        contraction_density,
        pronoun_density,
        question_density,
        exclamation_density,
        ellipsis_density,
        sent_len_var,
        uppercase_word_ratio,
        irregular_caps_ratio,
        code_keyword_density,
        code_symbol_density,
        function_def_flag,
        import_stmt_flag,
        code_line_ratio,
        code_likelihood,
        non_ascii_ratio,
        german_char_density,
        accented_char_density,
        is_german_indicator
    ]], dtype=np.float32)

    return raw_scalar, word_count, code_likelihood, is_german_indicator

def predict_text(text):
    text = str(text).strip()
    if not text:
        return {
            "prediction": "HUMAN",
            "probability": 0.0,
            "length_group": "very_short",
            "threshold": length_thresholds.get("very_short", 0.5),
            "code_likelihood": 0.0,
            "is_german": 0.0
        }

    lg = get_length_group(text)
    threshold = length_thresholds.get(lg, 0.50)

    # Transform TF-IDF
    X_word = word_vectorizer.transform([text])
    X_char = char_vectorizer.transform([text])
    X_mchar = multi_char_vectorizer.transform([text])

    # Embedding
    emb = transformer_model.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)

    # 26 Scalar features
    raw_scalar, word_count, code_likelihood, is_german = extract_features_single(text)
    X_scalar_scaled = linguistic_scaler.transform(raw_scalar)

    # Short interaction
    decay = np.array([[alpha_short_decay * np.exp(-word_count / 15.0)]], dtype=np.float32)
    X_short_inter = csr_matrix((emb * decay).astype(np.float32))

    # Combined feature matrix
    X_full = hstack([
        X_word,
        X_char,
        X_mchar,
        csr_matrix(emb),
        X_short_inter,
        csr_matrix(X_scalar_scaled)
    ], format="csr")

    prob = float(classifier.predict_proba(X_full)[:, 1][0])
    is_ai = prob >= threshold
    verdict = "AI" if is_ai else "HUMAN"

    return {
        "prediction": verdict,
        "probability": round(prob, 4),
        "length_group": lg,
        "threshold": round(threshold, 2),
        "code_likelihood": round(code_likelihood, 4),
        "is_german": round(is_german, 1)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "def count_digit(n):\n    while n != 0:\n        count += 1\n        n //= 10\n    return count"

    result = predict_text(input_text)
    print("=" * 60)
    print("V3.4 AI TEXT DETECTOR PREDICTION")
    print("=" * 60)
    print(f"Input Text: {repr(input_text[:80])}...")
    print(f"Prediction     : {result['prediction']}")
    print(f"Probability    : {result['probability']:.4f}")
    print(f"Length Group   : {result['length_group']}")
    print(f"Threshold      : {result['threshold']:.2f}")
    print(f"Code Likelihood: {result['code_likelihood']:.4f}")
    print(f"Is German      : {result['is_german']:.1f}")
    print("=" * 60)
