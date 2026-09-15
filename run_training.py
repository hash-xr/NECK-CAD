import logging
import torch
from torch.utils.data import DataLoader

from config.settings import SystemConfig
from main import NeckCADController
from training.dataset import create_patient_stratified_splits
from training.losses import MultiTaskFocalLoss
from training.trainer import MultiTaskTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NECK-CAD.RunTraining")


def main():
    config = SystemConfig()
    controller = NeckCADController(config=config)

    # 1. Load Pre-extracted Feature Dataset
    # Replace this with your dataset loading logic returning List[Dict]
    all_samples = []  # e.g., torch.load("data/processed_features.pt")

    if not all_samples:
        logger.warning("No samples loaded! Ensure 'all_samples' contains valid case data.")
        return

    # 2. Split Dataset by Patient ID
    train_ds, val_ds, test_ds = create_patient_stratified_splits(
        all_samples, val_size=0.2, test_size=0.1
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 3. Setup Loss, Optimizer, and Trainer
    criterion = MultiTaskFocalLoss()
    optimizer = torch.optim.AdamW(
        list(controller.fusion_engine.parameters()) + list(controller.classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-2,
    )

    trainer = MultiTaskTrainer(
        fusion_engine=controller.fusion_engine,
        classifier=controller.classifier,
        criterion=criterion,
        optimizer=optimizer,
        device=controller.device,
        checkpoint_dir="./checkpoints",
        audit_logger=controller.audit_logger,
        patience=5,  # Early stopping patience
    )

    # 4. Training Loop
    epochs = 30
    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} ---")
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        score = val_metrics["milan_macro_f1"]
        is_best = trainer.early_stopping(score)

        if is_best:
            trainer.save_checkpoint("best_neckcad_model.pt", val_metrics)

        if trainer.early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}. Training halted.")
            break


if __name__ == "__main__":
    main()