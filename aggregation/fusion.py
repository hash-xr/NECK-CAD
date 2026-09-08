import torch
import torch.nn as nn
import torch.nn.functional as F
from core.schema import DiagnosticResult, ImageMetadata, ImageStatus
from diagnosis.classifier import DiagnosticClassifier
from traceability.logger import AuditLogger

class MultiImageFusionEngine:
    def __init__(self, logger: AuditLogger):
        self.logger = logger

    def aggregate_session_predictions(
        self, 
        predictions_list: list[dict[str, torch.Tensor]], 
        valid_images: list[ImageMetadata]
    ) -> DiagnosticResult:
        """
        Fuses predictions across N valid images using attention-weighted pooling.
        """
        if not predictions_list:
            result = DiagnosticResult()
            result.primary_category = "Non-Diagnostic"
            result.milan_category = "Category I: Non-Diagnostic"
            result.specific_diagnosis = "Insufficient/No valid images processed"
            return result

        # Extract prob tensors across session images
        milan_probs_stack = torch.stack([p["milan_probs"].squeeze(0) for p in predictions_list]) # [N, Num_Milan]
        entity_probs_stack = torch.stack([p["entity_probs"].squeeze(0) for p in predictions_list]) # [N, Num_Entities]
        primary_probs_stack = torch.stack([p["primary_probs"].squeeze(0) for p in predictions_list]) # [N, Num_Primary]

        # Compute simple feature-weighting / quality-based scalar weights
        quality_scores = torch.tensor([max(img.quality_score, 1.0) for img in valid_images], dtype=torch.float32)
        attn_weights = F.softmax(quality_scores, dim=0).unsqueeze(1) # [N, 1]

        # Weighted pooling across session images
        fused_milan_probs = torch.sum(milan_probs_stack * attn_weights, dim=0)
        fused_entity_probs = torch.sum(entity_probs_stack * attn_weights, dim=0)
        fused_primary_probs = torch.sum(primary_probs_stack * attn_weights, dim=0)

        # Extract top diagnostic indices
        top_primary_idx = int(torch.argmax(fused_primary_probs).item())
        top_milan_idx = int(torch.argmax(fused_milan_probs).item())
        top_entity_idx = int(torch.argmax(fused_entity_probs).item())

        # Build differential diagnosis list
        top_entity_vals, top_entity_indices = torch.topk(
            fused_entity_probs, k=min(3, len(DiagnosticClassifier.SPECIFIC_ENTITIES))
        )
        differential = [
            (DiagnosticClassifier.SPECIFIC_ENTITIES[int(idx.item())], round(val.item(), 4))
            for idx, val in zip(top_entity_indices, top_entity_vals)
        ]

        # Map to DiagnosticResult data contract
        result = DiagnosticResult(
            primary_category=DiagnosticClassifier.PRIMARY_CATEGORIES[top_primary_idx],
            milan_category=DiagnosticClassifier.MILAN_CATEGORIES[top_milan_idx],
            specific_diagnosis=DiagnosticClassifier.SPECIFIC_ENTITIES[top_entity_idx],
            confidence_scores={
                "milan_confidence": round(fused_milan_probs[top_milan_idx].item(), 4),
                "entity_confidence": round(fused_entity_probs[top_entity_idx].item(), 4)
            },
            differential_diagnosis=differential
        )

        self.logger.log(
            f"Multi-image fusion complete across {len(predictions_list)} image(s). "
            f"Top diagnosis: {result.specific_diagnosis} ({result.milan_category})."
        )
        return result