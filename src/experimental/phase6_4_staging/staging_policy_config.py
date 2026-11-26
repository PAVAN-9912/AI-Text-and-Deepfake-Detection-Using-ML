"""
Staging Policy Configuration
============================
Defines the frozen decision thresholds, checkpoint paths, and cryptographic invariants
for the Phase 6.4 staging environment.

Derived strictly from the audited Phase 6.3.1 and Phase 6.3.2 specifications.
Thresholds are frozen and MUST NOT be modified or optimized at runtime.
"""

from pathlib import Path

PROJECT_ROOT = Path("C:/Users/PAVAN S/OneDrive/Desktop/AI_Content_Detection")

# -----------------------------------------------------------------------------
# CHECKPOINT INTEGRITY SPECIFICATIONS
# -----------------------------------------------------------------------------
VIT_CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "v12_vit" / "checkpoints" / "vit_deepfake_best.pth"
VIT_CHECKPOINT_SHA256 = "89df4bdb6689290d2638c6d6dafae19916b01ce451c49df7436d534be3c8631e"

EXPD_CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "phase6_2b" / "checkpoints" / "exp_d_finetuned_best_checkpoint.pth"
EXPD_CHECKPOINT_SHA256 = "e348707aa4e9aa9a1fb14da99d87d785695674b543ad3d9c88029ff6b9483529"

PROTECTED_TEST_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "phase6_1" / "PROTECTED_TEST_PROPOSED_MANIFEST.csv"
PROTECTED_TEST_MANIFEST_SHA256 = "e0be7470cc2c169a6127f4d9bd15c93469ca48239ff8c8821e4bd95a99a37a4c"

# -----------------------------------------------------------------------------
# POLICY C: EVIDENCE-GATED SPECIALIST FUSION THRESHOLDS
# -----------------------------------------------------------------------------
POLICY_C_CONFIG = {
    "vit_production_threshold": 0.475,
    "vit_fake_decisive_threshold": 0.60,
    "vit_real_decisive_threshold": 0.30,
    "exp_d_default_threshold": 0.50,
    "exp_d_ai_decisive_threshold": 0.70,
    "exp_d_real_decisive_threshold": 0.30,
}

# -----------------------------------------------------------------------------
# POLICY D: CONSERVATIVE FUSION THRESHOLDS
# -----------------------------------------------------------------------------
POLICY_D_CONFIG = {
    "vit_conservative_fake_threshold": 0.65,
    "vit_real_decisive_threshold": 0.30,
    "mutual_agreement_threshold": 0.50,
    "exp_d_faceless_ai_threshold": 0.85,
    "exp_d_faceless_real_threshold": 0.15,
    "exp_d_ai_conflict_threshold": 0.70,
}

# -----------------------------------------------------------------------------
# SYSTEM & RESOURCE LIMITS
# -----------------------------------------------------------------------------
MAX_CONCURRENT_INFERENCES = 1           # Strict single-worker to prevent CPU/RAM thrashing
MAX_QUEUE_CAPACITY = 5                  # Explicit bounded queue capacity limit
DEFAULT_INFERENCE_TIMEOUT_SEC = 60      # Max seconds per video evaluation
MAX_INPUT_FILE_SIZE_MB = 100            # Prevent OOM from excessively large videos
UNIFORM_SAMPLED_FRAMES = 8             # Exact Phase 6.2B/6.3 frame sampling parameter
FRAME_SIZE = (224, 224)                 # Standard model spatial input
