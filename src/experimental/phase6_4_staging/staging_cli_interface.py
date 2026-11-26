"""
Staging CLI & Test Interface
============================
Standalone interface for testing the Phase 6.4 two-expert video detection staging engine.

All outputs are formatted and explicitly labeled as EXPERIMENTAL STAGING RESULTS.
Does not alter or replace any production endpoints or API contracts.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("C:/Users/PAVAN S/OneDrive/Desktop/AI_Content_Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experimental.phase6_4_staging.staging_video_engine import StagingVideoEngine


def print_staging_result(result):
    print("=" * 80)
    print(" EXPERIMENTAL STAGING VIDEO DETECTION RESULT (PHASE 6.4) ")
    print(" NOTE: THIS IS A STAGING EXPERIMENT, NOT A PRODUCTION PREDICTION ")
    print("=" * 80)
    print(f"Input Video:             {result.input_video_path}")
    print(f"Staging Mode:            {result.staging_mode}")
    print(f"Selected Fusion Policy:  {result.policy_name}")
    print(f"Final Fusion Verdict:    {result.verdict.value}")
    print(f"Calibrated Fusion Score: {result.calibrated_score}")
    print(f"Decision Rule:           {result.decision_rule}")
    print(f"Decision Explanation:    {result.explanation}")
    print(f"Is Abstained:            {result.is_abstained}")
    print(f"Domain Conflict:         {result.domain_conflict}")
    print("-" * 80)
    print("EXPERT 1: PRODUCTION VISION TRANSFORMER (FACE SPECIALIST)")
    vit = result.face_vit_result
    print(f"  Status:                {vit.status.value}")
    print(f"  Face Detected:         {vit.face_detected}")
    print(f"  Fake Probability:      {vit.probability}")
    print(f"  Raw Prediction:        {vit.raw_prediction}")
    print(f"  Evidence Status:       {vit.evidence_status}")
    print(f"  Execution Time:        {vit.execution_time_ms:.2f} ms")
    print("-" * 80)
    print("EXPERT 2: CONVNEXT + HAAR WAVELET (FULL-FRAME FREQUENCY SPECIALIST)")
    exp = result.exp_d_result
    print(f"  Status:                {exp.status.value}")
    print(f"  AI Probability:        {exp.probability}")
    print(f"  Raw Prediction:        {exp.raw_prediction}")
    print(f"  Evidence Status:       {exp.evidence_status}")
    print(f"  Execution Time:        {exp.execution_time_ms:.2f} ms")
    print("-" * 80)
    print("PERFORMANCE & RESOURCE FOOTPRINT")
    print(f"  Total Staging Latency: {result.total_latency_ms:.2f} ms")
    print(f"  Peak Process RAM:      {result.peak_ram_mb:.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 6.4 Isolated Staging Video Fusion Engine")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--policy", choices=["POLICY_C", "POLICY_D"], default="POLICY_C", help="Fusion policy")
    parser.add_argument("--json", action="store_true", help="Output raw JSON result")
    args = parser.parse_args()

    engine = StagingVideoEngine()
    result = engine.analyze_video(args.video_path, policy=args.policy)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_staging_result(result)


if __name__ == "__main__":
    main()
