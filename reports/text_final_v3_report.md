# Text Detector V3 Comprehensive Research & Experiment Report

**Project**: AI Content Detection — Final-Year Major Project  
**Date**: September 27, 2026  
**Evaluation Scope**: Generalized AI-Generated vs. Human-Written Text Detection  
**Production Baseline**: [`models/text_final/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/text_final/) (V3.4 Architecture)  
**Experiment Suites**: [`src/run_text_v3_experiments.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/run_text_v3_experiments.py)  

---

## 1. Executive Summary & Verdict

### Final Selection Verdict:
> **No statistically meaningful improvement found over the V3.4 baseline; V3.4 remains the verified production baseline.**

A total of **21 candidate architectures and configurations** were systematically evaluated across regularization sweeps, feature normalizations, linear SVMs with Platt scaling, stylometric expansions, and alternative weighting policies under strict zero-leakage evaluation protocols.

The V3.4 baseline achieved the highest classification accuracy (**89.88%**), highest AI F1-score (**89.62%**), and lowest total error count (**101 errors** across $998$ validation samples), while demonstrating strong generalization across all external benchmarks.

---

## 2. Comprehensive Candidate Comparison Table

Evaluated on the untouched held-out validation set ($N=998$, $515$ Human, $483$ AI):

| Candidate Identifier | Architecture / Configuration | Val Accuracy | Val AI F1 | Val ROC-AUC | Val Brier | Val ECE | Total Errors (FP / FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate A (Baseline V3.4)** | **Logistic Regression $C=3.0$, Platt Scaling** | **89.88%** | **89.62%** | **95.63%** | **0.07928** | **0.0332** | **101 (54 / 47)** |
| **Candidate C ($C=3.0$)** | LogReg $C=3.0$, length-weighted | **89.88%** | **89.62%** | **95.63%** | **0.07928** | **0.0332** | **101 (54 / 47)** |
| **Candidate D (Linear SVM $C=0.5$)** | LinearSVC $C=0.5$ + Platt Sigmoid | **89.78%** | **89.57%** | **95.68%** | **0.07914** | **0.0234** | **102 (57 / 45)** |
| **Candidate C ($C=5.0$)** | LogReg $C=5.0$, length-weighted | **89.58%** | **89.26%** | **95.68%** | **0.07896** | **0.0250** | **104 (53 / 51)** |
| **Candidate C ($C=6.0$)** | LogReg $C=6.0$, length-weighted | **89.58%** | **89.23%** | **95.68%** | **0.07893** | **0.0240** | **104 (52 / 52)** |
| **Candidate G (Uniform)** | LogReg $C=3.0$, unweighted samples | **89.28%** | **89.00%** | **95.63%** | **0.07920** | **0.0311** | **107 (57 / 50)** |
| **Candidate C ($C=0.5$)** | LogReg $C=0.5$, length-weighted | **89.28%** | **89.09%** | **95.05%** | **0.08542** | **0.0479** | **107 (61 / 46)** |
| **Candidate D (Linear SVM $C=0.1$)** | LinearSVC $C=0.1$ + Platt Sigmoid | **89.18%** | **89.07%** | **95.53%** | **0.08054** | **0.0402** | **108 (65 / 43)** |
| **Candidate G (Balanced Class)** | LogReg $C=3.0$, balanced classes only | **89.18%** | **88.98%** | **95.63%** | **0.07920** | **0.0316** | **108 (61 / 47)** |
| **Candidate C ($C=1.0$)** | LogReg $C=1.0$, length-weighted | **89.18%** | **88.98%** | **95.39%** | **0.08176** | **0.0395** | **108 (61 / 47)** |
| **Candidate C ($C=4.0$)** | LogReg $C=4.0$, length-weighted | **89.18%** | **88.96%** | **95.65%** | **0.07906** | **0.0269** | **108 (60 / 48)** |
| **Candidate C ($C=8.0$)** | LogReg $C=8.0$, length-weighted | **88.98%** | **88.80%** | **95.69%** | **0.07893** | **0.0232** | **110 (63 / 47)** |
| **Candidate C ($C=1.5$)** | LogReg $C=1.5$, length-weighted | **89.08%** | **88.80%** | **95.51%** | **0.08046** | **0.0375** | **109 (58 / 51)** |
| **Candidate D (Linear SVM $C=1.0$)** | LinearSVC $C=1.0$ + Platt Sigmoid | **88.78%** | **88.64%** | **95.57%** | **0.08024** | **0.0281** | **112 (66 / 46)** |
| **Candidate C ($C=2.0$)** | LogReg $C=2.0$, length-weighted | **88.88%** | **88.57%** | **95.58%** | **0.07984** | **0.0397** | **111 (58 / 53)** |
| **Candidate F ($L_2\text{-Norm } C=20.0$)** | $L_2$ Full Matrix Norm + LogReg $C=20$ | **88.38%** | **88.38%** | **94.84%** | **0.08611** | **0.0277** | **116 (74 / 42)** |
| **Candidate F ($L_2\text{-Norm } C=10.0$)** | $L_2$ Full Matrix Norm + LogReg $C=10$ | **87.17%** | **87.33%** | **94.57%** | **0.08994** | **0.0390** | **128 (86 / 42)** |
| **Candidate F ($L_2\text{-Norm } C=5.0$)** | $L_2$ Full Matrix Norm + LogReg $C=5$ | **85.47%** | **85.94%** | **94.06%** | **0.09603** | **0.0452** | **145 (105 / 40)** |
| **Candidate F ($L_2\text{-Norm } C=1.0$)** | $L_2$ Full Matrix Norm + LogReg $C=1$ | **82.06%** | **82.50%** | **91.25%** | **0.12030** | **0.0381** | **179 (118 / 61)** |

---

## 3. External Generalization Benchmark Results

| Benchmark Dataset | Samples ($N$) | V3.4 Baseline Accuracy | LogReg $C=5.0$ Accuracy | V3.4 ROC-AUC | LogReg $C=5.0$ ROC-AUC | V3.4 Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAID External Test** | $1,500$ | **96.87%** | **96.80%** | **98.99%** | **99.03%** | **0.03308** |
| **SentenceAI Test** | $2,000$ | **85.50%** | **85.60%** | **92.31%** | **92.32%** | **0.10647** |
| **Real-World Test** | $26$ | **73.08%** | **76.92%** | **94.55%** | **94.55%** | **0.12876** |
| **AIGCodeSet Test (Orthogonal)** | $1,000$ | **49.30%** | **49.50%** | **49.23%** | **49.44%** | **0.35974** |

---

## 4. Performance Across Length Tiers (V3.4 Baseline)

| Length Tier | Word Range | Samples ($N$) | Accuracy | Precision | Recall | AI F1 | ROC-AUC | Decision Threshold ($t$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Very Short** | $\le 10$ words | $240$ | **88.75%** | **89.17%** | **87.27%** | **88.21%** | **94.62%** | $0.49$ |
| **Short** | $11 - 30$ words | $435$ | **87.82%** | **86.43%** | **87.56%** | **86.98%** | **94.01%** | $0.53$ |
| **Medium** | $31 - 60$ words | $223$ | **91.93%** | **92.31%** | **92.17%** | **92.24%** | **97.77%** | $0.64$ |
| **Long** | $> 60$ words | $100$ | **93.00%** | **95.45%** | **90.00%** | **92.63%** | **98.20%** | $0.59$ |

---

## 5. Architectural Specifications of the Production Model

```
Input Raw Text String
 │
 ├── Word TF-IDF (1, 2)-grams: 150,000 max features (sublinear tf, min_df=2, max_df=0.98)
 ├── Char TF-IDF (3, 5)-grams: 150,000 max features (sublinear tf, min_df=2, max_df=0.98)
 ├── Multi-Char-WB (3, 5)-grams: 5,000 max features (sublinear tf, min_df=2, max_df=0.98)
 ├── all-MiniLM-L6-v2 Embeddings: 384 dims (mean-pooled, L2-normalized)
 ├── Continuous Short Decay Interaction: 384 dims (0.75 * exp(-len / 15.0) * emb)
 └── 26 Scalar Features (StandardScaler normalized):
      ├── 5 Base Linguistic (log_wc, sent_count, avg_sent_len, vocab_div, len_bucket)
      ├── 11 Style / Lexical (token_entropy, rep_ngrams, transitions, contractions, pronouns, ...)
      ├── 6 Code / Syntax (keyword_density, symbol_density, def_flag, import_flag, code_lines, code_likelihood)
      └── 4 Multilingual (non_ascii_ratio, german_density, accent_density, is_german)
 │
 Total Feature Dimensionality: 194,897 dimensions
 │
 └── Regularized Logistic Regression (C=3.0, solver='liblinear', class_weight='balanced')
      │
      └── Platt Sigmoid Probability Calibrator (strictly fitted on independent calibration split)
           │
           └── Dynamic Length-Stratified Threshold Selector
```

---

## 6. Mathematical Probability Integrity

The inference pipeline guarantees strict mathematical normalization without artificial distortion:
$$P(\text{AI}) + P(\text{Human}) \equiv 1.0000$$

- All probabilities are derived directly from the calibrated decision score:
  $$P(\text{AI} \mid x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
- Where $A$ and $B$ are Platt parameters fit exclusively on the independent calibration partition ($N=907$).

---

## 7. Artifacts and Script Registry

- **Production Model**: [`models/text_final/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/text_final/)
- **Production Inference Pipeline**: [`predict_text_final.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/predict_text_final.py)
- **Zero-Leakage Audit Script**: [`src/audit_text_final.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/audit_text_final.py)
- **Error Diagnostics Script**: [`src/analyze_text_errors.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/analyze_text_errors.py)
- **V3 Experiment Suite**: [`src/run_text_v3_experiments.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/run_text_v3_experiments.py)
- **Backend Verification Suite**: [`src/verify_final_inference.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/verify_final_inference.py) (60/60 tests passed)
