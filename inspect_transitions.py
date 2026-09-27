import pandas as pd
import numpy as np

trans = pd.read_csv(r"models\text_final_v3_3\error_analysis\V32_VS_V33_ERROR_TRANSITIONS.csv")
trans["wc"] = trans["text"].apply(lambda x: len(str(x).split()))

print("=== 1. NEWLY INTRODUCED V3.3 FALSE POSITIVES (5) ===")
new_fp = trans[trans["transition"] == "V3.2_correct_to_V3.3_FP"]
for idx, r in new_fp.iterrows():
    p32, t32, p33, t33 = r["v32_probability"], r["v32_threshold"], r["v33_probability"], r["v33_threshold"]
    ds, lg, w = r["dataset"], r["length_group"], r["wc"]
    txt = r["text"][:100].replace("\n", " ")
    print(f"P32={p32:.4f} (t={t32:.2f}) -> P33={p33:.4f} (t={t33:.2f}) | {ds} | {lg} | wc={w}")
    print(f"  Text: {txt}\n")

print("=== 2. NEWLY INTRODUCED V3.3 FALSE NEGATIVES (6) ===")
new_fn = trans[trans["transition"] == "V3.2_correct_to_V3.3_FN"]
for idx, r in new_fn.iterrows():
    p32, t32, p33, t33 = r["v32_probability"], r["v32_threshold"], r["v33_probability"], r["v33_threshold"]
    ds, lg, w = r["dataset"], r["length_group"], r["wc"]
    txt = r["text"][:100].replace("\n", " ")
    print(f"P32={p32:.4f} (t={t32:.2f}) -> P33={p33:.4f} (t={t33:.2f}) | {ds} | {lg} | wc={w}")
    print(f"  Text: {txt}\n")

print("=== 3. V3.2 FP RESOLVED IN V3.3 (12) ===")
res_fp = trans[trans["transition"] == "V3.2_FP_to_V3.3_correct"]
for idx, r in res_fp.iterrows():
    p32, t32, p33, t33 = r["v32_probability"], r["v32_threshold"], r["v33_probability"], r["v33_threshold"]
    ds, lg, w = r["dataset"], r["length_group"], r["wc"]
    txt = r["text"][:100].replace("\n", " ")
    print(f"P32={p32:.4f} (t={t32:.2f}) -> P33={p33:.4f} (t={t33:.2f}) | {ds} | {lg} | wc={w}")
    print(f"  Text: {txt}\n")

print("=== 4. V3.2 FN RESOLVED IN V3.3 (4) ===")
res_fn = trans[trans["transition"] == "V3.2_FN_to_V3.3_correct"]
for idx, r in res_fn.iterrows():
    p32, t32, p33, t33 = r["v32_probability"], r["v32_threshold"], r["v33_probability"], r["v33_threshold"]
    ds, lg, w = r["dataset"], r["length_group"], r["wc"]
    txt = r["text"][:100].replace("\n", " ")
    print(f"P32={p32:.4f} (t={t32:.2f}) -> P33={p33:.4f} (t={t33:.2f}) | {ds} | {lg} | wc={w}")
    print(f"  Text: {txt}\n")
