import re
import numpy as np
import math

# Contractions
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

# Pronouns
pronouns = [
    r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b",
    r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b",
    r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b", r"\byourselves\b",
    r"\bthey\b", r"\bthem\b", r"\btheir\b", r"\btheirs\b", r"\bthemselves\b"
]
pronoun_regex = re.compile("|".join(pronouns), re.IGNORECASE)

# Transition words / expository markers
transition_words = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b", r"\bin addition\b",
    r"\bhowever\b", r"\btherefore\b", r"\bthus\b", r"\bconsequently\b",
    r"\bfirstly\b", r"\bsecondly\b", r"\bfinally\b", r"\bto summarize\b",
    r"\bon the other hand\b", r"\bas a result\b", r"\bin contrast\b"
]
transition_regex = re.compile("|".join(transition_words), re.IGNORECASE)

# Code keywords
code_kw_regex = re.compile(
    r"(\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:\(]|\breturn\b|\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import|\bwhile\s+.*:|\bfor\s+\w+\s+in\s+|\belif\s+.*:|\belse\s*:|\btry\s*:|\bexcept\s*.*:|\blambda\b|\byield\b|\bprint\s*\(|\bself\.\w+)",
    re.IGNORECASE
)

# Code operators
code_sym_regex = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\*\*|\b\d+\s*//\s*\d+\b|->|::|\{|\}|\[\s*\]|\(\s*\)|;\s*$)",
    re.MULTILINE
)

# German / Umlaut / Accented characters
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
    # Coefficient of variation (stabilized)
    return float(std_len / max(mean_len, 1.0))

def extract_v34_scalar_features(texts):
    features = []
    for text in texts:
        text = str(text)
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        wc_safe = max(word_count, 1)
        cc_safe = max(char_count, 1)

        # -------------------------------------------------------------
        # 1. BASE LINGUISTIC FEATURES (5)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 2. EXPANDED STYLE & LEXICAL FEATURES (9)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 3. CODE & SYNTAX FEATURES (6)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 4. MULTILINGUAL & NON-ENGLISH FEATURES (4)
        # -------------------------------------------------------------
        non_ascii_count = sum(1 for char in text if ord(char) > 127)
        non_ascii_ratio = non_ascii_count / cc_safe

        german_char_count = len(german_chars_regex.findall(text))
        german_char_density = german_char_count / cc_safe

        accented_char_count = len(accented_chars_regex.findall(text))
        accented_char_density = accented_char_count / cc_safe

        is_german_indicator = 1.0 if (german_char_count > 0 or bool(re.search(r"\b(der|die|das|und|ist|nicht|f[üu]r|ein|eine|einen|einer|vom|von|mit|sich|dem|den)\b", text, re.IGNORECASE))) else 0.0

        # Assemble full vector: 5 (base) + 11 (style) + 6 (code) + 4 (multilingual) = 26 features
        row = [
            log_word_count,
            sentence_count,
            avg_sentence_length,
            vocabulary_diversity,
            length_bucket,
            # Style (11)
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
            # Code (6)
            code_keyword_density,
            code_symbol_density,
            function_def_flag,
            import_stmt_flag,
            code_line_ratio,
            code_likelihood,
            # Multilingual (4)
            non_ascii_ratio,
            german_char_density,
            accented_char_density,
            is_german_indicator
        ]
        features.append(row)

    return np.asarray(features, dtype=np.float32)

test_samples = [
    ("Python Function", "def count_digit(n):\n    while n != 0:\n        count += 1\n        n //= 10\n    return count"),
    ("Formal Essay", "In conclusion, student designed projects have many benefits that can help them succeed in life. Furthermore, research supports this."),
    ("Conversational", "I'm really excited about what's coming, don't you think we'll win? What do you think?"),
    ("German Article", "Der Erfinder der E-Mail, Dr. Ray Tomlinson (74), ist am von einem Herzinfarkt gestorben. Tomlinson entwickelte das System.")
]

feats = extract_v34_scalar_features([s[1] for s in test_samples])
print("Extracted V3.4 Scalar Features shape:", feats.shape)
for (name, text), f_row in zip(test_samples, feats):
    print("=" * 60)
    print(f"Sample: {name}")
    print(f"  TokenEntropy: {f_row[5]:.3f} | TransRatio: {f_row[7]:.3f} | Contractions: {f_row[8]:.3f} | Pronouns: {f_row[9]:.3f}")
    print(f"  CodeKw: {f_row[16]:.3f} | CodeSym: {f_row[17]:.3f} | CodeLikelihood: {f_row[21]:.3f}")
    print(f"  NonAscii: {f_row[22]:.3f} | GermanChars: {f_row[23]:.3f} | IsGerman: {f_row[25]:.1f}")
