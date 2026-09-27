import re
import numpy as np

# Contraction regex
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

# Pronouns
pronouns = [
    r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b",
    r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b",
    r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b", r"\byourselves\b"
]
pronoun_regex = re.compile("|".join(pronouns), re.IGNORECASE)

# Code patterns
code_kw_regex = re.compile(
    r"(\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:\(]|\breturn\b|\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import|\bwhile\s+.*:|\bfor\s+\w+\s+in\s+|\belif\s+.*:|\bexcept\s*.*:|\blambda\s+.*:|\bprint\s*\(|\bself\.\w+)",
    re.IGNORECASE
)
code_sym_regex = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\b\d+\s*//\s*\d+\b|->|::|\{|\}|\[\s*\]|\(\s*\)|;\s*$)",
    re.MULTILINE
)

def extract_all_linguistic_and_code_features(texts):
    features = []
    for text in texts:
        text = str(text)
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        wc_safe = max(word_count, 1)
        cc_safe = max(char_count, 1)

        # --- Base 5 Linguistic Features ---
        sentence_count = sum(1 for char in text if char in ".!?")
        if sentence_count == 0:
            sentence_count = 1

        avg_sentence_length = word_count / max(sentence_count, 1)

        if word_count > 0:
            unique_words = len(set(word.lower() for word in words))
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

        # --- Style Features ---
        # 1. Contraction density
        contraction_count = len(contraction_regex.findall(text))
        contraction_density = contraction_count / wc_safe

        # 2. Pronoun density
        pronoun_count = len(pronoun_regex.findall(text))
        pronoun_density = pronoun_count / wc_safe

        # 3. Punctuation style
        question_mark_density = text.count('?') / wc_safe
        exclamation_density = len(re.findall(r'!(?!=)', text)) / wc_safe
        ellipsis_density = len(re.findall(r'\.{2,}', text)) / wc_safe

        # 4. Capitalization irregularity
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        uppercase_word_ratio = upper_words / wc_safe
        
        irreg_caps = sum(1 for w in words if any(c.isupper() for c in w[1:]) and not w.isupper())
        irregular_caps_ratio = irreg_caps / wc_safe

        # --- Code / Syntax Features ---
        code_kw_count = len(code_kw_regex.findall(text))
        code_keyword_density = code_kw_count / wc_safe

        code_sym_count = len(code_sym_regex.findall(text))
        code_symbol_density = code_sym_count / wc_safe

        function_def_flag = 1.0 if bool(re.search(r'\bdef\s+[a-zA-Z_]\w*\s*\(', text)) else 0.0
        import_stmt_flag = 1.0 if bool(re.search(r'(\bimport\s+[a-zA-Z_]|\bfrom\s+[a-zA-Z_]\w*\s+import)', text)) else 0.0
        has_indentation = 1.0 if bool(re.search(r'^\s{2,}\S', text, re.MULTILINE)) else 0.0

        code_likelihood = min(1.0, (code_keyword_density * 2.5 + code_symbol_density * 2.0 + function_def_flag * 0.4 + import_stmt_flag * 0.3 + has_indentation * 0.2))

        # Combined 17 linguistic, style, and code features
        features.append([
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
        ])

    return np.asarray(features, dtype=np.float32)

test_samples = [
    ("Code 1", "def count_digit(n):\n    while n != 0:\n        count += 1\n        n //= 10\n    return count"),
    ("Code 2", "def max_sum_list(lists):\n    return max(lists, key=sum)"),
    ("Prose 1", "If this happens, return to the previous step. Class participation is important."),
    ("Conversational", "I'm really excited about what's coming, don't you think we'll win?"),
    ("Formal Essay", "Self-confidence is really important for people to feel good about their own selves.")
]

feats = extract_all_linguistic_and_code_features([s[1] for s in test_samples])
print("Extracted feature shape:", feats.shape)
for (name, text), f_row in zip(test_samples, feats):
    print("=" * 60)
    print(f"Sample: {name}")
    print(f"  Contraction: {f_row[5]:.3f} | Pronoun: {f_row[6]:.3f} | Q: {f_row[7]:.3f} | Excl: {f_row[8]:.3f}")
    print(f"  CodeKw: {f_row[12]:.3f} | CodeSym: {f_row[13]:.3f} | FnDef: {f_row[14]:.1f} | CodeLikelihood: {f_row[16]:.3f}")
