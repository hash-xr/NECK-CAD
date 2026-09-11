import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NECK-CAD.Aggregation")


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning (MIL).
    Learns diagnostic attention weights across variable-length image feature sets.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.attention_v = nn.Linear(feature_dim, hidden_dim)
        self.attention_u = nn.Linear(feature_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Feature tensor of shape (N, feature_dim) for N FOVs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Raw attention logits of shape (N, 1)
                - Normalised softmax attention probabilities of shape (N, 1)
        """
        in_v = torch.tanh(self.attention_v(x))
        in_u = torch.sigmoid(self.attention_u(x))
        gated_attention = in_v * in_u  # Shape: (N, hidden_dim)

        logits = self.attention_weights(gated_attention)  # Shape: (N, 1)
        weights = F.softmax(logits, dim=0)  # Normalised across FOVs
        return logits, weights


class QualityWeightedAttentionFusion(nn.Module):
    """
    Combines learned neural attention scores with image quality scores
    to perform quality-gated feature aggregation across slide FOVs.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_pool = GatedAttentionPooling(feature_dim=feature_dim, hidden_dim=hidden_dim)
        logger.info(f"Initialised QualityWeightedAttentionFusion (feature_dim={feature_dim}).")

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates multiple FOV feature vectors into a single case-level feature embedding.

        Args:
            features (torch.Tensor): Tensor of shape (N, feature_dim) containing FOV embeddings.
            quality_scores (Optional[List[float]]): Quality gating scores [0, 1] for each FOV.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Aggregated case feature vector of shape (1, feature_dim)
                - Final per-FOV attention weight matrix of shape (N, 1)
        """
        n_fovs, dim = features.size()

        if n_fovs == 0:
            raise ValueError("Cannot perform feature fusion on an empty tensor collection.")

        # Single FOV pass-through optimization
        if n_fovs == 1:
            single_weight = torch.ones((1, 1), device=features.device, dtype=features.dtype)
            return features, single_weight

        # Compute neural attention scores across N FOVs
        _, raw_attention_weights = self.attention_pool(features)

        # Incorporate image quality score gating if provided
        if quality_scores is not None and len(quality_scores) == n_fovs:
            q_tensor = torch.tensor(
                quality_scores, device=features.device, dtype=features.dtype
            ).unsqueeze(1)
            # Modulate learned weights by image quality gating
            combined_weights = raw_attention_weights * q_tensor
            # Re-normalise across valid images
            weight_sum = torch.sum(combined_weights, dim=0, keepdim=True)
            if weight_sum.item() > 1e-6:
                final_weights = combined_weights / weight_sum
            else:
                final_weights = raw_attention_weights
        else:
            final_weights = raw_attention_weights

        # Compute quality-weighted attention pooling: (1, N) @ (N, dim) -> (1, dim)
        aggregated_embedding = torch.mm(final_weights.t(), features)
        return aggregated_embedding, final_weights