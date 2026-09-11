import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NECK-CAD.Classifier")


class MultiTaskHead(nn.Module):
    """
    Multi-task classification projection heads for Primary Category,
    Milan System Categories (I, II, III, IVA, IVB, V, VI), and Specific Entity Diagnosis.
    """

    def __init__(
        self,
        input_dim: int,
        num_primary: int = 3,
        num_milan: int = 7,  # Set to 7 to handle IVA and IVB explicitly
        num_entities: int = 10,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Common feature projection layer using LayerNorm (safe for B=1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Replaced BatchNorm1d to prevent batch size=1 crashes
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Task-Specific Projection Heads
        self.primary_head = nn.Linear(hidden_dim, num_primary)
        self.milan_head = nn.Linear(hidden_dim, num_milan)
        self.entity_head = nn.Linear(hidden_dim, num_entities)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Hybrid feature tensor of shape (B, input_dim).

        Returns:
            Dict[str, torch.Tensor]: Unnormalised logit dictionaries for all three tasks.
        """
        shared_features = self.shared_mlp(x)

        return {
            "primary_logits": self.primary_head(shared_features),
            "milan_logits": self.milan_head(shared_features),
            "entity_logits": self.entity_head(shared_features),
        }


class HybridDiagnosticClassifier(nn.Module):
    """
    Fuses deep learning feature embeddings with quantitative morphological features
    to output multi-task diagnostic logits.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        morphology_dim: int = 9,
        num_primary: int = 3,
        num_milan: int = 7,  # Updated to 7 categories
        num_entities: int = 10,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.morphology_dim = morphology_dim
        self.total_input_dim = embedding_dim + morphology_dim

        self.classifier_head = MultiTaskHead(
            input_dim=self.total_input_dim,
            num_primary=num_primary,
            num_milan=num_milan,
            num_entities=num_entities,
        )
        logger.info(
            f"Initialised HybridDiagnosticClassifier (total_input_dim={self.total_input_dim}, num_milan={num_milan})."
        )

    def forward(
        self,
        deep_features: torch.Tensor,
        morphology_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            deep_features (torch.Tensor): Tensor of shape (B, embedding_dim).
            morphology_features (Optional[torch.Tensor]): Tensor of shape (B, 9).

        Returns:
            Dict[str, torch.Tensor]: Diagnostic logits and predicted probabilities.
        """
        batch_size = deep_features.shape[0]

        if morphology_features is None:
            morphology_features = torch.zeros(
                (batch_size, self.morphology_dim),
                device=deep_features.device,
                dtype=deep_features.dtype,
            )

        # Concatenate Deep Embedding + Quantitative Morphology
        hybrid_features = torch.cat([deep_features, morphology_features], dim=1)

        # Forward through multi-task head
        logits_dict = self.classifier_head(hybrid_features)

        # Compute probabilities via Softmax
        probs_dict = {
            "primary_probs": F.softmax(logits_dict["primary_logits"], dim=-1),
            "milan_probs": F.softmax(logits_dict["milan_logits"], dim=-1),
            "entity_probs": F.softmax(logits_dict["entity_logits"], dim=-1),
        }

        return {**logits_dict, **probs_dict}