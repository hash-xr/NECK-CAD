import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NECK-CAD.Losses")


class MultiTaskFocalLoss(nn.Module):
    """
    Computes a weighted sum of focal losses for Primary Category,
    Milan Category, and Entity Diagnosis heads with class-imbalance weighting.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.weights = weights or {"primary": 0.25, "milan": 0.50, "entity": 0.25}
        self.class_weights = class_weights or {}
        self.gamma = gamma

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # 1. Primary Category Loss (3 classes)
        w_primary = self.class_weights.get("primary", None)
        if w_primary is not None:
            w_primary = w_primary.to(predictions["primary_logits"].device)
        losses["primary"] = self._focal_loss(
            predictions["primary_logits"], targets["primary"], weight=w_primary
        )

        # 2. Milan Category Loss (7 categories)
        w_milan = self.class_weights.get("milan", None)
        if w_milan is not None:
            w_milan = w_milan.to(predictions["milan_logits"].device)
        losses["milan"] = self._focal_loss(
            predictions["milan_logits"], targets["milan"], weight=w_milan
        )

        # 3. Entity Diagnosis Loss (10 entities - handles unlabelled targets marked -1)
        entity_targets = targets["entity"]
        valid_mask = entity_targets >= 0
        if valid_mask.any():
            w_entity = self.class_weights.get("entity", None)
            if w_entity is not None:
                w_entity = w_entity.to(predictions["entity_logits"].device)
            losses["entity"] = self._focal_loss(
                predictions["entity_logits"][valid_mask],
                entity_targets[valid_mask],
                weight=w_entity,
            )
        else:
            losses["entity"] = torch.tensor(0.0, device=predictions["primary_logits"].device)

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.weights["primary"] * losses["primary"]
            + self.weights["milan"] * losses["milan"]
            + self.weights["entity"] * losses["entity"]
        )
        losses["total"] = total_loss

        return losses