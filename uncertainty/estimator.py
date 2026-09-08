import torch
import torch.nn.functional as F
import numpy as np
from config.settings import SystemConfig
from core.schema import DiagnosticResult
from traceability.logger import AuditLogger

class UncertaintyEstimator:
    def __init__(self, config: SystemConfig, logger: AuditLogger):
        self.config = config
        self.logger = logger

    def calculate_entropy(self, probabilities: torch.Tensor) -> float:
        """Computes normalized Shannon entropy as a measure of predictive uncertainty."""
        # Add epsilon to prevent log(0)
        eps = 1e-12
        probs = torch.clamp(probabilities, min=eps, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs)).item()
        max_entropy = np.log(probs.shape[-1])
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def calculate_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> float:
        """
        Computes Free Energy score for Out-of-Distribution (OOD) detection.
        Lower energy indicates higher likelihood of in-distribution cytology data.
        """
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
        return float(energy.mean().item())

    def evaluate_uncertainty(
        self, 
        result: DiagnosticResult, 
        milan_probs: torch.Tensor, 
        raw_logits: torch.Tensor
    ) -> DiagnosticResult:
        """
        Calculates confidence bounds and energy-based OOD indicators.
        """
        # 1. Predictive Entropy Check
        milan_entropy = self.calculate_entropy(milan_probs)
        top_confidence = result.confidence_scores.get("milan_confidence", 0.0)

        if top_confidence < self.config.CONFIDENCE_THRESHOLD or milan_entropy > 0.75:
            result.is_uncertain = True
            self.logger.log(
                f"Prediction flagged as UNCERTAIN. Top confidence: {top_confidence:.2f}, Normalized Entropy: {milan_entropy:.2f}",
                level="WARNING"
            )

        # 2. Out-Of-Distribution (OOD) Free Energy Check
        energy_score = self.calculate_energy(raw_logits)
        if energy_score > self.config.OOD_ENERGY_THRESHOLD:
            result.is_ood = True
            self.logger.log(
                f"Input flagged as OUT-OF-DISTRIBUTION (OOD). Energy score ({energy_score:.2f}) exceeds threshold ({self.config.OOD_ENERGY_THRESHOLD:.2f}).",
                level="WARNING"
            )

        return result