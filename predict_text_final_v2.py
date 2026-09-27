import os
import re
import sys
import pickle
import math
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

# Base Directory & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "text_final_v2")

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

with open(os.path.join(MODEL_DIR, "probability_calibrator.pkl"), "rb") as f:
    calibrator_data = pickle.load(f)
    calibrator = calibrator_data["model"] if isinstance(calibrator_data, dict) else calibrator_data

with open(os.path.join(MODEL_DIR, "text_config.pkl"), "rb") as f:
    config = pickle.load(f)

transformer_name = config.get("transformer", "sentence-transformers/all-MiniLM-L6-v2")
transformer_model = SentenceTransformer(transformer_name)

length_thresholds = config.get("length_thresholds", {"very_short": 0.49, "short": 0.53, "medium": 0.64, "long": 0.59})
alpha_short_decay = config.get("short_interaction_alpha", 0.75)
is_enhanced = len(config.get("all_feature_names", [])) > 26

# Regex patterns
contractions = [
    r"can't", r"cannot", r"won't", r"don't", r"doesn't", r"didn't", r"isn't", r"aren't",
    r"wasn't", r"weren't", r"haven't", r"hasn't", r"hadn't", r"i'm", r"i've",
    r"i'll", r"i'd", r"you're", r"you've", r"you'll", r"you'd", r"he's",
    r"she's", r"it's", r"that's", r"we're", r"we've", r"we'll", r"they'd",
    r"they're", r"they've", r"they'll", r"couldn't", r"shouldn't", r"wouldn't",
    r"there's", r"what's", r"let's", r"who's"
]
contraction_regex = re.compile(r"\b(" + "|".join(c.replace("'", r"['\u2019]") for c in contractions) + r")\b", re.IGNORECASE)

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

STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
}

def calculate_token_entropy(words):
    if not words: return 0.0
    lengths = [len(w) for w in words]
    counts = Counter(lengths)
    total = len(lengths)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))

def calculate_char_entropy(text):
    if not text: return 0.0
    counts = Counter(text)
    total = len(text)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))

def calculate_rep_ngrams(words, n=3):
    if len(words) < n: return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return float((len(ngrams) - len(set(ngrams))) / max(len(ngrams), 1))

def calculate_sent_len_var(text, words):
    raw_s = [s.strip().split() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(raw_s) <= 1: return 0.0
    lens = [len(s) for s in raw_s]
    return float(np.std(lens) / max(np.mean(lens), 1.0))

def get_length_group(text):
    wc = len(str(text).split())
    if wc <= 10: return "very_short"
    elif wc <= 30: return "short"
    elif wc <= 60: return "medium"
    else: return "long"

def extract_features_single(text, enhanced=False):
    text = str(text).strip()
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    wc_safe = max(word_count, 1)
    cc_safe = max(char_count, 1)

    sentence_count = max(sum(1 for c in text if c in ".!?"), 1)
    avg_sentence_length = word_count / sentence_count
    vocab_diversity = len(set(w.lower() for w in words)) / wc_safe if word_count > 0 else 0.0
    log_word_count = np.log1p(word_count)

    if word_count <= 10: length_bucket = 0.0
    elif word_count <= 30: length_bucket = 1.0
    elif word_count <= 60: length_bucket = 2.0
    else: length_bucket = 3.0

    token_entropy = calculate_token_entropy(words)
    rep_ngram_ratio = calculate_rep_ngrams(words, n=3)
    transition_ratio = len(transition_regex.findall(text)) / wc_safe
    contraction_density = len(contraction_regex.findall(text)) / wc_safe
    pronoun_density = len(pronoun_regex.findall(text)) / wc_safe
    question_density = text.count("?") / wc_safe
    exclamation_density = len(re.findall(r"!(?!=)", text)) / wc_safe
    ellipsis_density = len(re.findall(r"\.{2,}", text)) / wc_safe
    sent_len_var = calculate_sent_len_var(text, words)

    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    uppercase_word_ratio = upper_words / wc_safe

    irreg_caps = sum(1 for w in words if any(c.isupper() for c in w[1:]) and not w.isupper())
    irregular_caps_ratio = irreg_caps / wc_safe

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

    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    non_ascii_ratio = non_ascii_count / cc_safe
    german_char_count = len(german_chars_regex.findall(text))
    german_char_density = german_char_count / cc_safe
    accented_char_count = len(accented_chars_regex.findall(text))
    accented_char_density = accented_char_count / cc_safe
    is_german_indicator = 1.0 if (german_char_count > 0 or bool(re.search(r"\b(der|die|das|und|ist|nicht|f[üu]r|ein|eine|einen|einer|vom|von|mit|sich|dem|den)\b", text, re.IGNORECASE))) else 0.0

    row = [
        log_word_count, sentence_count, avg_sentence_length, vocab_diversity, length_bucket,
        token_entropy, rep_ngram_ratio, transition_ratio, contraction_density, pronoun_density,
        question_density, exclamation_density, ellipsis_density, sent_len_var,
        uppercase_word_ratio, irregular_caps_ratio, code_keyword_density, code_symbol_density,
        function_def_flag, import_stmt_flag, code_line_ratio, code_likelihood,
        non_ascii_ratio, german_char_density, accented_char_density, is_german_indicator
    ]

    if enhanced:
        char_entropy = calculate_char_entropy(text)
        wlens = [len(w) for w in words] if words else [0]
        avg_word_len = float(np.mean(wlens))
        std_word_len = float(np.std(wlens)) if len(wlens) > 1 else 0.0
        stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
        stopword_ratio = stopword_count / wc_safe
        digit_count = sum(1 for c in text if c.isdigit())
        digit_density = digit_count / cc_safe
        punct_count = sum(1 for c in text if c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        punct_density = punct_count / cc_safe
        ttr_root = len(set(w.lower() for w in words)) / math.sqrt(wc_safe)
        hapax_count = sum(1 for _, cnt in Counter(w.lower() for w in words).items() if cnt == 1)
        hapax_ratio = hapax_count / wc_safe

        row.extend([
            char_entropy, avg_word_len, std_word_len, stopword_ratio,
            digit_density, punct_density, ttr_root, hapax_ratio
        ])

    raw_scalar = np.array([row], dtype=np.float32)
    return raw_scalar, word_count, code_likelihood, is_german_indicator

def predict_text(text):
    text = str(text).strip()
    if not text:
        return {
            "label": "Human Written",
            "ai_probability": 0.0,
            "human_probability": 1.0,
            "confidence": 1.0,
            "threshold": length_thresholds.get("very_short", 0.49),
            "word_count": 0,
            "length_group": "very_short",
            "model": "TextDetector_V2",
            "code_likelihood": 0.0,
            "is_german": 0.0
        }

    lg = get_length_group(text)
    threshold = length_thresholds.get(lg, 0.50)

    # Feature extraction
    X_word = word_vectorizer.transform([text])
    X_char = char_vectorizer.transform([text])
    X_mchar = multi_char_vectorizer.transform([text])
    emb = transformer_model.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    raw_scalar, word_count, code_likelihood, is_german = extract_features_single(text, enhanced=is_enhanced)
    X_scalar_scaled = linguistic_scaler.transform(raw_scalar)

    decay = np.array([[alpha_short_decay * np.exp(-word_count / 15.0)]], dtype=np.float32)
    X_short_inter = csr_matrix((emb * decay).astype(np.float32))

    X_full = hstack([
        X_word, X_char, X_mchar, csr_matrix(emb), X_short_inter, csr_matrix(X_scalar_scaled)
    ], format="csr")

    # Decision score & Platt calibrated probability
    decision_score = classifier.decision_function(X_full).reshape(-1, 1)
    calibrated_probs = calibrator.predict_proba(decision_score)[0]
    ai_prob = float(calibrated_probs[1])
    human_prob = float(calibrated_probs[0])

    # Enforce strict normalization
    prob_sum = ai_prob + human_prob
    ai_prob = ai_prob / prob_sum
    human_prob = 1.0 - ai_prob

    is_ai = ai_prob >= threshold
    label = "AI Generated" if is_ai else "Human Written"
    confidence = ai_prob if is_ai else human_prob

    return {
        "label": label,
        "ai_probability": round(ai_prob, 4),
        "human_probability": round(human_prob, 4),
        "confidence": round(confidence, 4),
        "threshold": round(threshold, 2),
        "word_count": word_count,
        "length_group": lg,
        "model": "TextDetector_V2",
        "code_likelihood": round(code_likelihood, 4),
        "is_german": round(is_german, 1)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "In conclusion, student designed projects have many benefits that can help them succeed in life."

    res = predict_text(input_text)
    print("=" * 60)
    print("TEXT DETECTOR V2 INFERENCE")
    print("=" * 60)
    print(f"Input Text        : {repr(input_text[:80])}...")
    print(f"Label             : {res['label']}")
    print(f"AI Probability    : {res['ai_probability']*100:.2f}%")
    print(f"Human Probability : {res['human_probability']*100:.2f}%")
    print(f"Confidence        : {res['confidence']*100:.2f}%")
    print(f"Word Count        : {res['word_count']}")
    print(f"Length Group      : {res['length_group']}")
    print(f"Threshold         : {res['threshold']:.2f}")
    print(f"Model             : {res['model']}")
    print("=" * 60)
