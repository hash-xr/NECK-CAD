import os
import logging
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

from traceability.logger import AuditLogger
from training.losses import MultiTaskFocalLoss

logger = logging.getLogger("NECK-CAD.Trainer")

MILAN_CATEGORY_NAMES = [
    "I: Non-Diagnostic",
    "II: Non-Neoplastic",
    "III: AUS",
    "IVA: Neoplasm Benign",
    "IVB: SUMP",
    "V: Suspicious",
    "VI: Malignant",
]


class EarlyStopping:
    """Monitors validation metric and triggers early stopping when progress stalls."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class MultiTaskTrainer:
    """
    Manages optimization loops, mixed precision, gradient clipping, early stopping,
    detailed Milan metric tracking, and audit trail logging.
    """

    def __init__(
        self,
        fusion_engine: nn.Module,
        classifier: nn.Module,
        criterion: MultiTaskFocalLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        audit_logger: Optional[AuditLogger] = None,
        max_grad_norm: float = 1.0,
        patience: int = 5,
    ):
        self.fusion_engine = fusion_engine.to(device)
        self.classifier = classifier.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.audit_logger = audit_logger
        self.max_grad_norm = max_grad_norm

        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    def _log_audit(self, message: str):
        if self.audit_logger:
            self.audit_logger.log(message)
        logger.info(message)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.fusion_engine.train()
        self.classifier.train()

        running_losses = {"total": 0.0, "primary": 0.0, "milan": 0.0, "entity": 0.0}
        total_batches = 0

        for batch in dataloader:
            embeddings = batch["embeddings"].squeeze(0).to(self.device)  # [N, Dim]
            morphology = batch["morphology"].squeeze(0).to(self.device)  # [N, 9]
            quality_scores = batch["quality_scores"].squeeze(0).tolist()

            targets = {
                "primary": batch["primary"].to(self.device),
                "milan": batch["milan"].to(self.device),
                "entity": batch["entity"].to(self.device),
            }

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                case_morphology = torch.mean(morphology, dim=0, keepdim=True)
                case_embedding, _ = self.fusion_engine(embeddings, quality_scores=quality_scores)
                preds = self.classifier(case_embedding, case_morphology)
                loss_dict = self.criterion(preds, targets)

            # Mixed-Precision Backward Pass with Gradient Clipping
            self.scaler.scale(loss_dict["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.fusion_engine.parameters()) + list(self.classifier.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in running_losses:
                running_losses[key] += loss_dict[key].item()
            total_batches += 1

        if self.scheduler:
            self.scheduler.step()

        epoch_losses = {k: v / max(1, total_batches) for k, v in running_losses.items()}
        self._log_audit(
            f"Epoch Training Complete | Total Loss: {epoch_losses['total']:.4f} | "
            f"Milan Loss: {epoch_losses['milan']:.4f} | Primary Loss: {epoch_losses['primary']:.4f}"
        )
        return epoch_losses

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.fusion_engine.eval()
        self.classifier.eval()

        milan_preds, milan_targets = [], []
        primary_preds, primary_targets = [], []

        for batch in dataloader:
            embeddings = batch["embeddings"].squeeze(0).to(self.device)
            morphology = batch["morphology"].squeeze(0).to(self.device)
            quality_scores = batch["quality_scores"].squeeze(0).tolist()

            case_morphology = torch.mean(morphology, dim=0, keepdim=True)
            case_embedding, _ = self.fusion_engine(embeddings, quality_scores=quality_scores)
            outputs = self.classifier(case_embedding, case_morphology)

            milan_preds.append(int(torch.argmax(outputs["milan_probs"], dim=1).cpu()))
            milan_targets.append(int(batch["milan"]))

            primary_preds.append(int(torch.argmax(outputs["primary_probs"], dim=1).cpu()))
            primary_targets.append(int(batch["primary"]))

        # Compute Detailed Classification Report for Milan Risk Categories
        report = classification_report(
            milan_targets,
            milan_preds,
            target_names=MILAN_CATEGORY_NAMES[: len(set(milan_targets + milan_preds))],
            output_dict=True,
            zero_division=0,
        )

        macro_f1 = f1_score(milan_targets, milan_preds, average="macro", zero_division=0)
        milan_acc = accuracy_score(milan_targets, milan_preds)
        primary_acc = accuracy_score(primary_targets, primary_preds)

        eval_results = {
            "milan_macro_f1": float(macro_f1),
            "milan_accuracy": float(milan_acc),
            "primary_accuracy": float(primary_acc),
            "detailed_milan_report": report,
        }

        self._log_audit(
            f"Validation Results | Milan Macro F1: {macro_f1:.4f} | "
            f"Milan Accuracy: {milan_acc:.4f} | Primary Accuracy: {primary_acc:.4f}"
        )

        return eval_results

    def save_checkpoint(self, filename: str, metrics: Dict[str, Any]):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "fusion_state_dict": self.fusion_engine.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        self._log_audit(f"Checkpoint saved successfully to '{path}'.")