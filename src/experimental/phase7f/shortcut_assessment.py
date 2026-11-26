"""
Source and Codec Shortcut Forensic Assessment for Phase 7F.3

Investigates whether decoded 224x224 RGB frames retain systematic forensic artifacts
correlated with source repository, original codec, frame rate, or generator family.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import f_oneway, kruskal

from .preprocessing import StandardVideoPreprocessor, PreprocessingConfig


def compute_frame_statistics(frames_array: np.ndarray) -> Dict[str, float]:
    """
    Computes spatial and temporal signal metrics on (16, 224, 224, 3) uint8 RGB frames:
    - laplacian_var: variance of Laplacian (measure of high-frequency sharpness / compression blur)
    - sobel_gradient_mean: average edge gradient
    - inter_frame_diff_mean: mean L1 difference between consecutive frames (temporal velocity)
    - color_variance_mean: spatial color variation
    """
    T, H, W, C = frames_array.shape
    lap_vars = []
    sobel_grads = []
    diffs = []

    for t in range(T):
        gray = cv2.cvtColor(frames_array[t], cv2.COLOR_RGB2GRAY)
        # Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_vars.append(float(lap.var()))

        # Sobel gradient
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        sobel_grads.append(float(np.mean(mag)))

        if t > 0:
            diff = np.abs(frames_array[t].astype(np.float32) - frames_array[t - 1].astype(np.float32))
            diffs.append(float(np.mean(diff)))

    return {
        "laplacian_var_mean": float(np.mean(lap_vars)),
        "sobel_gradient_mean": float(np.mean(sobel_grads)),
        "inter_frame_diff_mean": float(np.mean(diffs)) if diffs else 0.0,
    }


def run_source_codec_assessment(
    manifest_path: str,
    project_root: str,
    output_report_path: Optional[str] = None,
    max_samples_per_cohort: int = 25,
) -> Dict[str, Any]:
    """
    Audits the development partition (train + val) for residual codec/source shortcuts
    and constructs stratification and matched evaluation tables.
    """
    df = pd.read_csv(manifest_path)
    # Strictly inspect development data (do not evaluate test)
    dev_df = df[df["partition"].isin(["train", "val"])].copy().reset_index(drop=True)

    preprocessor = StandardVideoPreprocessor()
    records = []

    print(f"[Shortcut Assessment] Analyzing {len(dev_df)} development videos...")

    # Sample balanced subset for intensive signal analysis to benchmark runtime
    grouped = dev_df.groupby(["category", "partition"])
    sampled_indices = []
    for _, grp in grouped:
        n = min(len(grp), max_samples_per_cohort)
        sampled_indices.extend(grp.sample(n=n, random_state=42).index.tolist())

    audit_subset = dev_df.loc[sampled_indices].copy()
    print(f"[Shortcut Assessment] Selected {len(audit_subset)} representative development clips.")

    for idx, row in audit_subset.iterrows():
        fp = row["filepath"]
        if not os.path.isabs(fp):
            fp = os.path.join(project_root, fp)

        frames, status = preprocessor.decode_video(fp)
        if not status["success"] or frames is None:
            continue

        stats = compute_frame_statistics(frames)
        stats.update(
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "partition": row["partition"],
                "ground_truth_class": row["ground_truth_class"],
                "source_dataset": row["source_dataset"],
                "codec": row["codec"],
                "original_width": row["width"],
                "original_height": row["height"],
                "fps": row["fps"],
                "duration_sec": row["duration_sec"],
                "generator_or_method": row["generator_or_method"],
            }
        )
        records.append(stats)

    res_df = pd.DataFrame(records)

    # 1. Stratification by Original Codec
    codec_strat = res_df.groupby(["codec", "ground_truth_class"])[
        ["laplacian_var_mean", "sobel_gradient_mean", "inter_frame_diff_mean"]
    ].agg(["count", "mean", "std"])

    # 2. Stratification by Source Dataset
    source_strat = res_df.groupby(["source_dataset", "ground_truth_class"])[
        ["laplacian_var_mean", "sobel_gradient_mean", "inter_frame_diff_mean"]
    ].agg(["count", "mean", "std"])

    # 3. Stratification by Generator
    gen_strat = res_df.groupby(["generator_or_method", "ground_truth_class"])[
        ["laplacian_var_mean", "inter_frame_diff_mean"]
    ].agg(["count", "mean", "std"])

    # 4. Statistical Tests across Codecs
    codecs_present = res_df["codec"].unique()
    kruskal_lap = None
    if len(codecs_present) > 1:
        groups = [res_df[res_df["codec"] == c]["laplacian_var_mean"].values for c in codecs_present]
        kruskal_lap = kruskal(*groups)

    # 5. Matched Source/Codec Subsets Identification
    # In Celeb-DF v2: C2 (genuine) vs C6 (synthesis) are both FMP4 codec in AVI/MP4 container!
    fmp4_real = dev_df[(dev_df["category"] == "C2_real_face_man") & (dev_df["codec"] == "FMP4")]
    fmp4_fake = dev_df[(dev_df["category"] == "C6_ai_face_man") & (dev_df["codec"] == "FMP4")]

    matched_subsets = {
        "matched_fmp4_celebdf": {
            "description": "Celeb-DF v2 matched codec (FMP4) across Real and DeepFaceLab",
            "genuine_count": len(fmp4_real),
            "ai_count": len(fmp4_fake),
            "train_genuine": len(fmp4_real[fmp4_real["partition"] == "train"]),
            "train_ai": len(fmp4_fake[fmp4_fake["partition"] == "train"]),
            "val_genuine": len(fmp4_real[fmp4_real["partition"] == "val"]),
            "val_ai": len(fmp4_fake[fmp4_fake["partition"] == "val"]),
            "fair_codec_match": True,
        },
        "matched_h264_cohort": {
            "description": "H.264 codec representation across Real and AI",
            "genuine_count": len(dev_df[(dev_df["ground_truth_class"] == "genuine") & (dev_df["codec"] == "h264")]),
            "ai_count": len(dev_df[(dev_df["ground_truth_class"] == "ai_generated") & (dev_df["codec"] == "h264")]),
            "fair_codec_match": False,
            "limitation_note": "CRITICAL LIMITATION: Zero genuine development videos use H.264 (all genuine outdoor/faceless videos are WebM VP8/VP9/AV1/Theora from Wikimedia). A fair H.264 matched comparison across nature scenes is physically impossible with current dataset.",
        },
        "matched_webm_cohort": {
            "description": "WebM (VP8/VP9/AV1) codec representation across Real and AI",
            "genuine_count": len(dev_df[(dev_df["ground_truth_class"] == "genuine") & (dev_df["codec"].isin(["VP90", "VP80", "AV01", "theo"]))]),
            "ai_count": len(dev_df[(dev_df["ground_truth_class"] == "ai_generated") & (dev_df["codec"].isin(["VP90", "VP80", "AV01", "theo"]))]),
            "fair_codec_match": False,
            "limitation_note": "CRITICAL LIMITATION: Zero AI-generated development videos use WebM/VP9 (all generative benchmark models export MP4/H.264). A fair WebM matched comparison across AI generators is physically impossible with current dataset.",
        },
    }

    report = {
        "num_clips_audited": len(res_df),
        "codec_stratification": codec_strat,
        "source_stratification": source_strat,
        "generator_stratification": gen_strat,
        "kruskal_test_laplacian_across_codecs": {
            "statistic": float(kruskal_lap.statistic) if kruskal_lap else None,
            "p_value": float(kruskal_lap.pvalue) if kruskal_lap else None,
        },
        "matched_subsets": matched_subsets,
    }

    return report
