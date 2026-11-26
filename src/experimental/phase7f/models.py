"""
ST-SeqNet: Spatio-Temporal Sequence Network for Anti-Shortcut Video Forensics (Phase 7F.3)

Architecture Specification:
1. Spatial Backbone:
   - EfficientNet-B0 (feature extractor).
   - Extracts a D=1280 dimensional spatial feature vector per frame.
   - Operates across all T=16 frames in parallel (reshaping [B*T, 3, 224, 224] -> [B*T, 1280]).
2. Spatial Feature-Difference Representation:
   - For consecutive feature vectors f_1, f_2, ..., f_T:
   - Computes temporal velocity: Delta f_t = f_t - f_{t-1} for t >= 2, Delta f_1 = 0.
   - Concatenates [f_t, Delta f_t] -> dimension 2560.
   - Projects via Linear(2560, 256) + LayerNorm + GELU.
   - Encodes both instantaneous visual quality and inter-frame temporal consistency / blending jitter.
3. Lightweight Bidirectional GRU:
   - Input dimension: 256.
   - Hidden dimension: 128 (bidirectional -> output dimension 256).
   - 1 layer, batch_first=True, dropout=0.0.
4. Temporal Attention Mechanism:
   - Learns time-step relevance weights alpha_t over the 16 temporal steps:
     score_t = v^T tanh(W_att * h_t + b_att)
     alpha = softmax(scores)
   - Computes context vector c = sum(alpha_t * h_t) in R^256.
   - Allows forensic inspection of which frames triggered the detection.
5. Binary Classification Head:
   - MLP: Linear(256, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 1).
   - Produces raw logit and sigmoid probability (0.0 = Real/Genuine, 1.0 = AI-Generated).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple


class TemporalAttention(nn.Module):
    """
    Self-attention pooling across temporal frames.
    Input: [B, T, D] -> Output context: [B, D], weights: [B, T]
    """

    def __init__(self, feature_dim: int, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        scores = self.projection(x)  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        context = torch.sum(x * weights, dim=1)  # [B, D]
        return context, weights.squeeze(-1)  # [B, D], [B, T]


class STSeqNet(nn.Module):
    """
    Spatio-Temporal Sequence Network (ST-SeqNet).
    CPU and CUDA compatible.
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        gru_hidden_dim: int = 128,
        projection_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 1. Spatial Backbone: EfficientNet-B0
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        )
        base_model = models.efficientnet_b0(weights=weights)

        # Retain feature extractor up to adaptive pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool  # Output: [B*T, 1280, 1, 1]
        self.backbone_dim = 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # 2. Spatial Feature-Difference Representation
        # Concatenation of [f_t, Delta f_t] gives 1280 * 2 = 2560 dimensions
        self.feature_diff_projection = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_output_dim = gru_hidden_dim * 2  # 256

        # 4. Temporal Attention
        self.attention = TemporalAttention(
            feature_dim=self.gru_output_dim, attention_dim=128
        )

        # 5. Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] -> returns spatial features: [B, T, 1280]
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = self.features(x_reshaped)
        pooled = self.avgpool(feats)  # [B*T, 1280, 1, 1]
        pooled = torch.flatten(pooled, 1)  # [B*T, 1280]
        return pooled.view(B, T, self.backbone_dim)  # [B, T, 1280]

    def compute_feature_differences(self, f: torch.Tensor) -> torch.Tensor:
        """
        Computes temporal velocity vectors Delta f_t = f_t - f_{t-1}.
        f: [B, T, D] -> returns projected representation: [B, T, 256]
        """
        B, T, D = f.shape

        # Compute delta: for t=0, delta is zero vector
        delta_0 = torch.zeros(B, 1, D, device=f.device, dtype=f.dtype)
        deltas_rest = f[:, 1:, :] - f[:, :-1, :]  # [B, T-1, D]
        deltas = torch.cat([delta_0, deltas_rest], dim=1)  # [B, T, D]

        # Concatenate [f_t, Delta f_t] -> [B, T, 2*D]
        combined = torch.cat([f, deltas], dim=-1)
        projected = self.feature_diff_projection(combined)  # [B, T, 256]
        return projected

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input video tensor [B, T, C, H, W] or [B, 16, 3, 224, 224]
        Returns:
            Dict containing:
                'logits': [B, 1] raw unbounded logit
                'probabilities': [B, 1] sigmoid probability in [0, 1]
                'attention_weights': [B, T] temporal attention distribution
                'context_vector': [B, 256] temporal pooled representation
        """
        # 1. Spatial feature extraction across all frames
        spatial_feats = self.extract_spatial_features(x)  # [B, 16, 1280]

        # 2. Temporal feature difference representation
        seq_inputs = self.compute_feature_differences(spatial_feats)  # [B, 16, 256]

        # 3. Bidirectional GRU
        gru_out, _ = self.gru(seq_inputs)  # [B, 16, 256]

        # 4. Temporal Attention Pooling
        context, attn_weights = self.attention(gru_out)  # [B, 256], [B, 16]

        # 5. Classification
        logits = self.classifier(context)  # [B, 1]
        probs = torch.sigmoid(logits)  # [B, 1]

        return {
            "logits": logits,
            "probabilities": probs,
            "attention_weights": attn_weights,
            "context_vector": context,
        }
