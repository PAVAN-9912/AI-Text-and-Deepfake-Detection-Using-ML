"""
Staging Fusion Router
=====================
Implements the audited Policy C (Evidence-Gated Specialist Fusion) and Policy D
(Conservative Fusion) decision algorithms.

Guarantees:
- Pure functional logic based solely on inference-time expert outputs.
- Complete independence from ground-truth or test metadata.
- Safe handling of missing, NaN, infinite, or error expert states.
"""

import math
from typing import Optional
import logging

from .staging_result_schema import (
    ExpertInferenceResult,
    FusionDecisionResult,
    FusionVerdict,
    ExpertStatus
)
from .staging_policy_config import POLICY_C_CONFIG, POLICY_D_CONFIG

logger = logging.getLogger("staging_fusion_router")


class StagingFusionRouter:
    """Pure decision router executing audited staging fusion policies."""

    @staticmethod
    def _is_valid_prob(p: Optional[float]) -> bool:
        return p is not None and not math.isnan(p) and not math.isinf(p) and 0.0 <= p <= 1.0

    @classmethod
    def evaluate(
        cls,
        face_vit_result: ExpertInferenceResult,
        exp_d_result: ExpertInferenceResult,
        policy: str = "POLICY_C",
        total_latency_ms: float = 0.0,
        peak_ram_mb: float = 0.0,
        input_video_path: str = ""
    ) -> FusionDecisionResult:
        """
        Evaluates the two expert outputs under the requested policy.
        Supported policies: "POLICY_C", "POLICY_D".
        """
        pol_key = policy.upper()
        if pol_key not in ("POLICY_C", "POLICY_D"):
            raise ValueError(f"Unsupported fusion policy '{policy}'. Must be POLICY_C or POLICY_D.")

        # Robustness Check 1: Missing or error states
        if face_vit_result.status == ExpertStatus.ERROR or exp_d_result.status == ExpertStatus.ERROR:
            err_msg = f"Inference error detected: Face ViT status={face_vit_result.status.value}, Exp D status={exp_d_result.status.value}"
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name=pol_key,
                verdict=FusionVerdict.INCONCLUSIVE,
                raw_prediction=-1,
                calibrated_score=None,
                decision_rule="RULE_ERROR_FALLBACK",
                explanation=err_msg,
                is_abstained=True,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Robustness Check 2: Invalid probabilities
        p_exp = exp_d_result.probability
        p_vit = face_vit_result.probability

        if not cls._is_valid_prob(p_exp):
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name=pol_key,
                verdict=FusionVerdict.INCONCLUSIVE,
                raw_prediction=-1,
                calibrated_score=None,
                decision_rule="RULE_INVALID_SCORE_FALLBACK",
                explanation=f"Exp D produced an invalid probability: {p_exp}",
                is_abstained=True,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Execute selected policy
        if pol_key == "POLICY_C":
            return cls._evaluate_policy_c(
                face_vit_result, exp_d_result, total_latency_ms, peak_ram_mb, input_video_path
            )
        else:
            return cls._evaluate_policy_d(
                face_vit_result, exp_d_result, total_latency_ms, peak_ram_mb, input_video_path
            )

    @classmethod
    def _evaluate_policy_c(
        cls,
        face_vit_result: ExpertInferenceResult,
        exp_d_result: ExpertInferenceResult,
        total_latency_ms: float,
        peak_ram_mb: float,
        input_video_path: str
    ) -> FusionDecisionResult:
        p_exp = exp_d_result.probability
        p_vit = face_vit_result.probability
        has_face = face_vit_result.face_detected and face_vit_result.status != ExpertStatus.INSUFFICIENT_EVIDENCE

        # Case 1: Faceless video (Face ViT has no face evidence)
        if not has_face:
            if p_exp >= POLICY_C_CONFIG["exp_d_default_threshold"]:
                verdict = FusionVerdict.AI_GENERATED
                raw_pred = 1
                rule = "C1_FACELESS_AI_AUTHORITY"
                expl = f"Faceless video: Delegated to Exp D full-frame AI authority (P_ai={p_exp:.4f} >= {POLICY_C_CONFIG['exp_d_default_threshold']})"
            else:
                verdict = FusionVerdict.LIKELY_HUMAN
                raw_pred = 0
                rule = "C1_FACELESS_GENUINE_AUTHORITY"
                expl = f"Faceless video: Delegated to Exp D full-frame Genuine authority (P_ai={p_exp:.4f} < {POLICY_C_CONFIG['exp_d_default_threshold']})"
            
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_C",
                verdict=verdict,
                raw_prediction=raw_pred,
                calibrated_score=p_exp,
                decision_rule=rule,
                explanation=expl,
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # For face-present videos, ensure p_vit is valid
        if not cls._is_valid_prob(p_vit):
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_C",
                verdict=FusionVerdict.INCONCLUSIVE,
                raw_prediction=-1,
                calibrated_score=None,
                decision_rule="RULE_INVALID_SCORE_FALLBACK",
                explanation=f"Face ViT produced an invalid probability: {p_vit}",
                is_abstained=True,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Case 2: Decisive Face ViT FAKE evidence (FaceSwap / facial manipulation)
        if p_vit >= POLICY_C_CONFIG["vit_fake_decisive_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_C",
                verdict=FusionVerdict.AI_GENERATED,
                raw_prediction=1,
                calibrated_score=p_vit,
                decision_rule="C2_DECISIVE_FACIAL_MANIPULATION",
                explanation=f"Face ViT decisive facial manipulation evidence (P_fake={p_vit:.4f} >= {POLICY_C_CONFIG['vit_fake_decisive_threshold']})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Case 3: Decisive Face ViT REAL evidence (Authentic facial features)
        if p_vit <= POLICY_C_CONFIG["vit_real_decisive_threshold"]:
            if p_exp >= POLICY_C_CONFIG["exp_d_ai_decisive_threshold"]:
                # Direct domain conflict: Strong facial genuine vs strong full-frame wavelet AI
                return FusionDecisionResult(
                    staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                    policy_name="POLICY_C",
                    verdict=FusionVerdict.INCONCLUSIVE,
                    raw_prediction=-1,
                    calibrated_score=None,
                    decision_rule="C3_STRONG_DOMAIN_CONFLICT",
                    explanation=f"Strong domain conflict: Face ViT authentic face (P_fake={p_vit:.4f}) vs Exp D strong AI (P_ai={p_exp:.4f}) -> INCONCLUSIVE",
                    is_abstained=True,
                    domain_conflict=True,
                    face_vit_result=face_vit_result,
                    exp_d_result=exp_d_result,
                    total_latency_ms=total_latency_ms,
                    peak_ram_mb=peak_ram_mb,
                    input_video_path=input_video_path
                )
            else:
                return FusionDecisionResult(
                    staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                    policy_name="POLICY_C",
                    verdict=FusionVerdict.LIKELY_HUMAN,
                    raw_prediction=0,
                    calibrated_score=p_vit,
                    decision_rule="C3_DECISIVE_AUTHENTIC_FACE",
                    explanation=f"Face ViT authentic facial evidence (P_fake={p_vit:.4f} <= {POLICY_C_CONFIG['vit_real_decisive_threshold']}) without full-frame conflict",
                    is_abstained=False,
                    domain_conflict=False,
                    face_vit_result=face_vit_result,
                    exp_d_result=exp_d_result,
                    total_latency_ms=total_latency_ms,
                    peak_ram_mb=peak_ram_mb,
                    input_video_path=input_video_path
                )

        # Case 4: Face ViT in equivocal zone (0.30 < P_fake < 0.60)
        if p_exp >= POLICY_C_CONFIG["exp_d_ai_decisive_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_C",
                verdict=FusionVerdict.AI_GENERATED,
                raw_prediction=1,
                calibrated_score=p_exp,
                decision_rule="C4_EQUIVOCAL_VIT_RESOLVED_BY_EXPD_AI",
                explanation=f"Equivocal Face ViT (P_fake={p_vit:.4f}) resolved by decisive Exp D AI evidence (P_ai={p_exp:.4f} >= {POLICY_C_CONFIG['exp_d_ai_decisive_threshold']})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )
        elif p_exp <= POLICY_C_CONFIG["exp_d_real_decisive_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_C",
                verdict=FusionVerdict.LIKELY_HUMAN,
                raw_prediction=0,
                calibrated_score=p_exp,
                decision_rule="C4_EQUIVOCAL_VIT_RESOLVED_BY_EXPD_GENUINE",
                explanation=f"Equivocal Face ViT (P_fake={p_vit:.4f}) resolved by decisive Exp D Genuine evidence (P_ai={p_exp:.4f} <= {POLICY_C_CONFIG['exp_d_real_decisive_threshold']})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )
        else:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_C",
                verdict=FusionVerdict.INCONCLUSIVE,
                raw_prediction=-1,
                calibrated_score=None,
                decision_rule="C4_EQUIVOCAL_DUAL_EXPERT_CONFLICT",
                explanation=f"Equivocal evidence from both experts: Face ViT P_fake={p_vit:.4f}, Exp D P_ai={p_exp:.4f} -> INCONCLUSIVE",
                is_abstained=True,
                domain_conflict=True,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

    @classmethod
    def _evaluate_policy_d(
        cls,
        face_vit_result: ExpertInferenceResult,
        exp_d_result: ExpertInferenceResult,
        total_latency_ms: float,
        peak_ram_mb: float,
        input_video_path: str
    ) -> FusionDecisionResult:
        p_exp = exp_d_result.probability
        p_vit = face_vit_result.probability
        has_face = face_vit_result.face_detected and face_vit_result.status != ExpertStatus.INSUFFICIENT_EVIDENCE

        # Case 1: Faceless video
        if not has_face:
            if p_exp >= POLICY_D_CONFIG["exp_d_faceless_ai_threshold"]:
                verdict = FusionVerdict.AI_GENERATED
                raw_pred = 1
                rule = "D1_CONSERVATIVE_FACELESS_AI"
                expl = f"Conservative faceless: Very high Exp D AI confidence (P_ai={p_exp:.4f} >= {POLICY_D_CONFIG['exp_d_faceless_ai_threshold']})"
                is_abs = False
                score = p_exp
            elif p_exp <= POLICY_D_CONFIG["exp_d_faceless_real_threshold"]:
                verdict = FusionVerdict.LIKELY_HUMAN
                raw_pred = 0
                rule = "D1_CONSERVATIVE_FACELESS_GENUINE"
                expl = f"Conservative faceless: Very low Exp D AI confidence (P_ai={p_exp:.4f} <= {POLICY_D_CONFIG['exp_d_faceless_real_threshold']})"
                is_abs = False
                score = p_exp
            else:
                verdict = FusionVerdict.INCONCLUSIVE
                raw_pred = -1
                rule = "D1_CONSERVATIVE_FACELESS_ABSTAIN"
                expl = f"Conservative faceless: Intermediate Exp D score (P_ai={p_exp:.4f}) -> ABSTAIN"
                is_abs = True
                score = None

            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_D",
                verdict=verdict,
                raw_prediction=raw_pred,
                calibrated_score=score,
                decision_rule=rule,
                explanation=expl,
                is_abstained=is_abs,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        if not cls._is_valid_prob(p_vit):
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_D",
                verdict=FusionVerdict.INCONCLUSIVE,
                raw_prediction=-1,
                calibrated_score=None,
                decision_rule="RULE_INVALID_SCORE_FALLBACK",
                explanation=f"Face ViT produced an invalid probability: {p_vit}",
                is_abstained=True,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Case 2: Decisive high-margin facial manipulation
        if p_vit >= POLICY_D_CONFIG["vit_conservative_fake_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_D",
                verdict=FusionVerdict.AI_GENERATED,
                raw_prediction=1,
                calibrated_score=p_vit,
                decision_rule="D2_DECISIVE_FACIAL_MANIPULATION",
                explanation=f"Conservative: Decisive Face ViT facial manipulation (P_fake={p_vit:.4f} >= {POLICY_D_CONFIG['vit_conservative_fake_threshold']})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Case 3: Mutual Agreement across 0.50 boundary
        if p_vit >= POLICY_D_CONFIG["mutual_agreement_threshold"] and p_exp >= POLICY_D_CONFIG["mutual_agreement_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_D",
                verdict=FusionVerdict.AI_GENERATED,
                raw_prediction=1,
                calibrated_score=max(p_vit, p_exp),
                decision_rule="D3_MUTUAL_AI_AGREEMENT",
                explanation=f"Conservative: Mutual AI agreement (ViT={p_vit:.4f}, ExpD={p_exp:.4f})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        if p_vit < POLICY_D_CONFIG["mutual_agreement_threshold"] and p_exp < POLICY_D_CONFIG["mutual_agreement_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_D",
                verdict=FusionVerdict.LIKELY_HUMAN,
                raw_prediction=0,
                calibrated_score=min(p_vit, p_exp),
                decision_rule="D3_MUTUAL_GENUINE_AGREEMENT",
                explanation=f"Conservative: Mutual Genuine agreement (ViT={p_vit:.4f}, ExpD={p_exp:.4f})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Case 4: Confident Face ViT Real
        if p_vit <= POLICY_D_CONFIG["vit_real_decisive_threshold"] and p_exp < POLICY_D_CONFIG["exp_d_ai_conflict_threshold"]:
            return FusionDecisionResult(
                staging_mode="EXPERIMENTAL_STAGING_RESULTS",
                policy_name="POLICY_D",
                verdict=FusionVerdict.LIKELY_HUMAN,
                raw_prediction=0,
                calibrated_score=p_vit,
                decision_rule="D4_CONFIDENT_FACE_VIT_GENUINE",
                explanation=f"Conservative: Strong Face ViT genuine face (P_fake={p_vit:.4f} <= {POLICY_D_CONFIG['vit_real_decisive_threshold']})",
                is_abstained=False,
                domain_conflict=False,
                face_vit_result=face_vit_result,
                exp_d_result=exp_d_result,
                total_latency_ms=total_latency_ms,
                peak_ram_mb=peak_ram_mb,
                input_video_path=input_video_path
            )

        # Case 5: All conflicts / equivocal cases
        return FusionDecisionResult(
            staging_mode="EXPERIMENTAL_STAGING_RESULTS",
            policy_name="POLICY_D",
            verdict=FusionVerdict.INCONCLUSIVE,
            raw_prediction=-1,
            calibrated_score=None,
            decision_rule="D5_CONSERVATIVE_CONFLICT_ABSTAIN",
            explanation=f"Conservative: Conflicting or equivocal evidence (ViT={p_vit:.4f}, ExpD={p_exp:.4f}) -> ABSTAIN",
            is_abstained=True,
            domain_conflict=True,
            face_vit_result=face_vit_result,
            exp_d_result=exp_d_result,
            total_latency_ms=total_latency_ms,
            peak_ram_mb=peak_ram_mb,
            input_video_path=input_video_path
        )
