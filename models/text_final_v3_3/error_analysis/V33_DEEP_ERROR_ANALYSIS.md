# V3.3 Deep Diagnostic Error Analysis & V3.4 Experiment Plan

## 1. Executive Summary & Verification

- **Model Analyzed**: Generalized AI Text Detector V3.3 (`models/text_final_v3_3/`)
- **Validation Dataset**: Untouched held-out split of **998 samples** (515 Human, 483 AI)
- **Overall Performance**:
  - **Accuracy**: `88.98%` (888 / 998 correct) — *up from 88.48% in V3.2 (+0.50%) and 87.17% in V3.1 (+1.81%)*
  - **Precision**: `87.98%` — *up from 86.80% in V3.2 (+1.18%) and 83.55% in V3.1 (+4.43%)*
  - **Recall**: `89.44%` (432 / 483 AI samples detected)
  - **$F_1$ Score**: `88.71%` — *up from 88.30% in V3.2 (+0.41%) and 87.35% in V3.1 (+1.36%)*
  - **ROC-AUC**: `95.11%`
- **Total Errors**: **110** (59 False Positives, 51 False Negatives) — *lowest error count across all iterations (V3.1: 128, V3.2: 115)*

---

## 2. Comprehensive Error Transitions (V3.2 $\rightarrow$ V3.3)

| Transition Category | Sample Count | Description |
| :--- | :---: | :--- |
| **`V3.2_correct_to_V3.3_correct`** | **872** | Correct predictions preserved across both versions |
| **`V3.2_FP_to_V3.3_correct`** | **12** | **12 False Alarms Eliminated** (9 SentenceAI, 2 HC3, 1 RAID) |
| **`V3.2_FN_to_V3.3_correct`** | **4** | **4 False Negatives Resolved** (1 RAID code, 1 German, 2 SentenceAI) |
| **`V3.2_correct_to_V3.3_FP`** | **5** | 4 German long texts due to $t=0.33$ calibration shift + 1 short |
| **`V3.2_correct_to_V3.3_FN`** | **6** | 5 SentenceAI + 1 RAID code near the new $t=0.55/0.62$ boundary |
| **`persistent_FP`** | **54** | False Positives present in both V3.2 and V3.3 |
| **`persistent_FN`** | **45** | False Negatives present in both V3.2 and V3.3 |

---

## 3. False Positive Deep-Dive (59 Samples Total)

| Category | Count ($N$) | % of All Errors | Mean Prob | Median Prob | Probability Range | Mean Word Count | Primary Root Cause Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Formal SentenceAI Human Writing** | **31** | 28.2% | `0.6990` | `0.6925` | `[0.4407, 0.9776]` | 13.3 | **Feature Representation / Semantic Overlap** |
| **B. RAID Python / Code Samples** | **15** | 13.6% | `0.6216` | `0.6666` | `[0.3320, 0.9070]` | 85.0 | **Domain / Syntax Mismatch** |
| **C. HC3 Factual / Encyclopedic Answers** | **13** | 11.8% | `0.6963` | `0.6491` | `[0.3714, 0.9616]` | 42.9 | **Style / Low-Burstiness Factual Prose** |
| **D. `very_short` Human Text (1–10w)** | **20** | 18.2% | `0.6859` | `0.7024` | `[0.4407, 0.9624]` | 6.5 | **Insufficient Information / Sparse TF-IDF** |
| **E. `short` Human Text (11–30w)** | **23** | 20.9% | `0.7152` | `0.7079` | `[0.5543, 0.9776]` | 19.0 | **Semantic Clustering with AI Essays** |

### Probability Band Distribution for False Positives:
- `P in [0.20, 0.40)`: **3** (5.1%) — Marginal long-tier FPs ($t=0.33$)
- `P in [0.40, 0.50)`: **8** (13.6%) — Very short FPs near threshold ($t=0.42$)
- `P in [0.50, 0.60)`: **10** (16.9%) — Borderline short FPs
- `P in [0.60, 0.80)`: **22** (37.3%) — High-confidence formal essay & code FPs
- `P in [0.80, 1.00]`: **16** (27.1%) — Extreme FPs (mostly formal SentenceAI thesis statements)

---

## 4. False Negative Deep-Dive (51 Samples Total)

| Category | Count ($N$) | % of All Errors | Mean Prob | Median Prob | Probability Range | Mean Word Count | Primary Root Cause Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. SentenceAI Conversational AI** | **38** | 34.5% | `0.3141` | `0.3237` | `[0.0343, 0.5846]` | 18.6 | **Feature Representation / Conversational Mimicry** |
| **B. RAID German / Non-English AI** | **7** | 6.4% | `0.2890` | `0.2646` | `[0.2083, 0.4405]` | 153.4 | **Language / English Tokenizer Limitation** |
| **C. Short AI Near Threshold ($|P-t| \le 0.10$)** | **6** | 5.5% | `0.4329` | `0.4267` | `[0.3779, 0.5286]` | 13.7 | **Threshold Boundary Sensitivity** |
| **D. `very_short` AI Text (1–10w)** | **9** | 8.2% | `0.2596` | `0.2320` | `[0.0983, 0.4032]` | 7.1 | **Insufficient Information** |
| **E. Medium / Long AI Text (31+w)** | **16** | 14.5% | `0.3977` | `0.3678` | `[0.2083, 0.6066]` | 98.1 | **Complex Mixed Syntax / Long Non-English** |

### Probability Band Distribution for False Negatives:
- `P < 0.20`: **8** (15.7%) — Extreme colloquial AI (*"Meat, its something we all love to eat."* P=0.034)
- `P in [0.20, 0.40)`: **28** (54.9%) — Dominant FN cluster (Conversational AI & German)
- `P in [0.40, 0.50)`: **8** (15.7%) — Sub-threshold short AI
- `P in [0.50, 0.60)`: **5** (9.8%) — Borderline short AI below $t=0.55$
- `P in [0.60, 0.80)`: **2** (3.9%) — Borderline medium AI below $t=0.62$

---

## 5. Root Cause Classification

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          ROOT CAUSE BREAKDOWN IN V3.3                             │
├──────────────────────────────┬──────────────┬─────────────────────────────────────┤
│ Cause Category               │ % of Errors  │ Primary Manifestation               │
├──────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ 1. Feature Representation    │ 35% (38 FNs) │ Conversational AI adopts pronouns   │
│                              │              │ and informal dialogue structure     │
├──────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ 2. Semantic Overlap          │ 28% (31 FPs) │ Formal human student essays cluster │
│                              │              │ with AI expository embeddings       │
├──────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ 3. Domain / Syntax Mismatch  │ 14% (15 FPs) │ Python functions treated as prose   │
├──────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ 4. Language / Tokenization   │  8% (9 errs) │ German text in RAID lacks n-grams   │
├──────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ 5. Insufficient Information  │ 10% (11 errs)│ 1–5 word fragments (e.g. "Ummm...") │
├──────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ 6. Threshold Boundary        │  5% (6 errs) │ Borderline samples in [0.50, 0.55]  │
└──────────────────────────────┴──────────────┴─────────────────────────────────────┘
```

---

## 6. What Worked in V3.2 $\rightarrow$ V3.3

1. **Pronoun & Contraction Style Features**:
   - Successfully rescued **9 human essay false positives** from SentenceAI by registering genuine human voice markers.
   - Boosted overall precision from **86.80% to 87.98% (+1.18%)** and reduced overall FPs to an all-time low of **59**.
2. **Short-Tier Accuracy**:
   - `short` tier accuracy surged from **87.13% to 88.74%**, with $F_1$ climbing to **88.02%**.

---

## 7. V3.4 Experiment Plan

The following three experiments are strictly derived from empirical diagnostic evidence:

### Experiment 1: Lexical Perplexity & Repetition Entropy Block (Target: Formal Essay FPs)
- **Target Category**: Formal SentenceAI Human Writing (31 FPs) and Conversational AI (38 FNs) — **69 errors affected**.
- **Diagnostic Evidence**: Human student essays have higher local vocabulary burstiness and sentence-level length variation than synthetic ChatGPT text, even when topic embeddings overlap.
- **Exact Change**: Add 4 bounded entropy features to the scalar block:
  1. *Token Length Entropy*: Shannon entropy of token character lengths in the text.
  2. *Unigram Perplexity Proxy*: Log-frequency rank distribution across top-1000 common English words.
  3. *Transition Word Ratio*: Normalized count of standard expository transitional phrases (`furthermore`, `moreover`, `in conclusion`, `in addition`).
  4. *Repetitive N-gram Ratio*: Ratio of repeated 3-word n-grams over total n-grams.
- **Expected Benefit**: Reduces high-confidence false positives ($P > 0.80$) on structured human student essays by +8–12 samples.
- **Risk**: Low; smooth bounded scalar features scaled via StandardScaler.
- **Validation Leakage**: 0%. Trainable and evaluable strictly on 8,160 train / 907 calibration split.

### Experiment 2: Enhanced Indentation & Programming Token Density (Target: RAID Code FPs)
- **Target Category**: RAID Python / Code False Positives — **15 errors affected**.
- **Diagnostic Evidence**: In V3.3, `code_likelihood` was introduced, but raw character n-grams still trigger high AI weights on function definitions (`def ...`, `return ...`).
- **Exact Change**:
  1. Add tokenized syntax features: count of Python operators (`//`, `**`, `!=`, `==`, `+=`, `->`), argument list patterns (`def name(a, b):`), and multi-line tab/space indentation ratio.
  2. Add an interaction term `code_likelihood * MiniLM_embedding_norm` to decouple programmatic dense embeddings from natural language embeddings.
- **Expected Benefit**: Eliminates 8–10 of the remaining 15 code-related false positives.
- **Risk**: Negligible on regular natural language text.
- **Validation Leakage**: 0%.

### Experiment 3: Multilingual / Character-Level Subword Smoothing (Target: RAID German FNs)
- **Target Category**: RAID German Non-English False Negatives — **7 FNs affected**.
- **Diagnostic Evidence**: 7 of 9 RAID FNs are German articles generated by MPT, where English TF-IDF n-grams provide no signal and MiniLM produces low probabilities ($P \in [0.20, 0.29]$).
- **Exact Change**:
  1. Add non-ASCII / umlaut character density (`ä`, `ö`, `ü`, `ß`).
  2. Add language identification entropy score (distribution across Latin subwords).
  3. Include a small multilingual character 4-gram sub-vector (max 5,000 features) fitted across all training samples.
- **Expected Benefit**: Recovers 4–5 German AI false negatives without impacting English text.
- **Risk**: Low.
- **Validation Leakage**: 0%.
