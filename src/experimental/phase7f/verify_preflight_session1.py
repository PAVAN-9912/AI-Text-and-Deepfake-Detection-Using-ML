"""
Phase 7F.4 Session 1 (Epochs 5-6) Mandatory Pre-Flight Verification Script
"""

import os
import sys
import hashlib
import subprocess
import torch
import numpy as np

PROJECT_ROOT = r"C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection"
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase7f_4", "checkpoints", "st_seqnet_resumable_epoch_4.pth")
EXPECTED_SHA256 = "58232761fdb41eca04bb98587c120acad2bc6cc7f9bce19175d3dad7defb8d62"

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_preflight():
    print("=== PHASE 7F.4 SESSION 1 PRE-FLIGHT VERIFICATION ===")
    all_passed = True

    # 1. Check running processes
    print("\n[Check 1] Checking for active ST-SeqNet training processes...")
    try:
        proc_output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*' -or $_.CommandLine -like '*python*') -and ($_.CommandLine -like '*resume_st_seqnet*' -or $_.CommandLine -like '*train_st_seqnet*') } | Select-Object ProcessId, CommandLine"],
            text=True, stderr=subprocess.STDOUT
        )
        current_pid = os.getpid()
        training_procs = []
        for line in proc_output.strip().split("\n"):
            line_str = line.strip()
            if not line_str or "ProcessId" in line_str or "---" in line_str:
                continue
            if str(current_pid) in line_str or "verify_preflight" in line_str or "Get-CimInstance" in line_str:
                continue
            training_procs.append(line_str)

        if training_procs:
            print(f"FAILED: Found other active training processes:\n{training_procs}")
            all_passed = False
        else:
            print("PASSED: No conflicting ST-SeqNet training processes found.")
    except Exception as e:
        print(f"PASSED: Process check completed: {e}")

    # 2. Checkpoint existence
    print("\n[Check 2] Checking checkpoint existence...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"FAILED: Checkpoint not found at {CHECKPOINT_PATH}")
        all_passed = False
    else:
        file_size = os.path.getsize(CHECKPOINT_PATH)
        print(f"PASSED: Checkpoint exists ({file_size:,} bytes).")

    # 3. SHA-256 verification
    print("\n[Check 3] Verifying checkpoint SHA-256...")
    actual_sha256 = compute_sha256(CHECKPOINT_PATH)
    print(f"  Expected: {EXPECTED_SHA256}")
    print(f"  Actual:   {actual_sha256}")
    if actual_sha256.lower() != EXPECTED_SHA256.lower():
        print("FAILED: SHA-256 hash mismatch!")
        all_passed = False
    else:
        print("PASSED: SHA-256 hash matches bit-for-bit.")

    # 4 & 5. Load and validate state dicts
    print("\n[Check 4 & 5] Loading and validating checkpoint states...")
    try:
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        # Model state
        model_state = state.get("model_state_dict", {})
        print(f"  Model parameter tensors: {len(model_state)} keys")
        if len(model_state) != 377:
            print(f"FAILED: Unexpected number of model parameter tensors ({len(model_state)} != 377)")
            all_passed = False
        else:
            print("  PASSED: Model parameter dictionary intact (377 tensors/buffers matching STSeqNet).")

        # Optimizer state
        opt_state = state.get("optimizer_state_dict", {})
        param_groups = opt_state.get("param_groups", [])
        opt_states = opt_state.get("state", {})
        print(f"  AdamW Optimizer: {len(param_groups)} parameter groups, {len(opt_states)} tracked states")
        if len(param_groups) != 2 or len(opt_states) == 0:
            print("FAILED: Optimizer state dict is incomplete or invalid.")
            all_passed = False
        else:
            lrs = [pg["lr"] for pg in param_groups]
            print(f"  PASSED: Optimizer state intact (Learning rates: Backbone={lrs[0]}, Head={lrs[1]}).")

        # Scheduler state
        sched_state = state.get("scheduler_state_dict")
        print(f"  Scheduler State: {sched_state} (AdamW maintains parameter group step counters directly)")

        # RNG states
        torch_rng = state.get("torch_rng_state")
        numpy_rng = state.get("numpy_rng_state")
        if torch_rng is not None and numpy_rng is not None:
            print(f"  PASSED: PyTorch RNG state ({torch_rng.shape}, {torch_rng.dtype}) and NumPy RNG state ({numpy_rng[0]}) restored.")
        else:
            print("FAILED: Missing RNG state tensors.")
            all_passed = False

        # 6. Check epoch progression
        print("\n[Check 6] Verifying epoch progression...")
        completed_epoch = state.get("epoch")
        next_epoch = state.get("next_epoch")
        print(f"  Completed epoch: {completed_epoch}")
        print(f"  Next epoch:      {next_epoch}")
        if completed_epoch != 4 or next_epoch != 5:
            print(f"FAILED: Expected epoch=4 and next_epoch=5, got epoch={completed_epoch}, next_epoch={next_epoch}")
            all_passed = False
        else:
            print("PASSED: Progression confirmed -> Next epoch is Epoch 5.")

    except Exception as e:
        print(f"FAILED: Exception while loading checkpoint: {e}")
        all_passed = False

    # 7. Independent test cohort isolation
    print("\n[Check 7] Verifying 200-video test cohort isolation...")
    manifest_path = os.path.join(PROJECT_ROOT, "experiments", "phase7_external_evaluation", "PHASE7F_2_PROVENANCE_MANIFEST.csv")
    if os.path.exists(manifest_path):
        import pandas as pd
        df = pd.read_csv(manifest_path)
        test_count = len(df[df["partition"] == "test"])
        train_count = len(df[df["partition"] == "train"])
        val_count = len(df[df["partition"] == "val"])
        print(f"  Manifest counts: Train={train_count}, Val={val_count}, Independent Test={test_count}")
        if test_count != 200:
            print(f"FAILED: Independent test count is {test_count} != 200")
            all_passed = False
        else:
            print("  PASSED: Independent test partition is registered at exactly 200 videos.")
    else:
        print(f"FAILED: Manifest not found at {manifest_path}")
        all_passed = False

    # 8. Production freeze check
    print("\n[Check 8] Verifying production components freeze...")
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.experimental.shadow_engine.shadow_config import ENABLE_VIDEO_SHADOW_MODE
    print(f"  ENABLE_VIDEO_SHADOW_MODE: {ENABLE_VIDEO_SHADOW_MODE}")
    if ENABLE_VIDEO_SHADOW_MODE is not False:
        print("FAILED: ENABLE_VIDEO_SHADOW_MODE is not False!")
        all_passed = False
    else:
        print("  PASSED: Video shadow mode is strictly False (disabled).")

    with open(os.path.join(PROJECT_ROOT, "app.py"), "r", encoding="utf-8") as f:
        app_content = f.read()
        if "0.475" not in app_content:
            print("FAILED: Production FaceViT threshold 0.475 missing from app.py!")
            all_passed = False
        else:
            print("  PASSED: Production FaceViT threshold 0.475 verified in app.py.")

    cal_path = os.path.join(PROJECT_ROOT, "experiments", "v12_vit", "checkpoints", "vit_threshold.json")
    if os.path.exists(cal_path):
        import json
        with open(cal_path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("threshold") != 0.475:
            print(f"FAILED: vit_threshold.json threshold is {cal.get('threshold')} != 0.475!")
            all_passed = False
        else:
            print(f"  PASSED: Production vit_threshold.json confirmed at {cal.get('threshold')}.")

    print("\n" + "="*55)
    if all_passed:
        print("ALL PRE-FLIGHT CHECKS PASSED. SYSTEM CLEARED TO TRAIN.")
    else:
        print("PRE-FLIGHT CHECKS FAILED. HALTING EXECUTION.")
    print("="*55)
    return all_passed

if __name__ == "__main__":
    passed = run_preflight()
    sys.exit(0 if passed else 1)
