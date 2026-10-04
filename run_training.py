# run_training.py
import os
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
    # Ensure checkpoint folders exist ahead of time
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    controller = NeckCADController(config=config)

    # 1. FIX: Load Pre-extracted Feature Dataset from cache
    processed_data_path = "data/processed_features.pt"
    
    if os.path.exists(processed_data_path):
        logger.info(f"Loading cached patient features from '{processed_data_path}'...")
        all_samples = torch.load(processed_data_path)
    else:
        logger.error(
            f"Feature cache map not found at '{processed_data_path}'! "
            f"You must run 'extract_features.py' first to build the case vectors."
        )
        return

    if not all_samples:
        logger.warning("No samples loaded! Ensure 'all_samples' contains valid case data.")
        return

    # 2. Split Dataset by Patient ID using configuration variables
    random_seed = getattr(config, "RANDOM_STATE", 42)
    train_ds, val_ds, test_ds = create_patient_stratified_splits(
        all_samples, 
        val_size=0.2, 
        test_size=0.1,
        random_state=random_seed
    )

    # Dataset batch sizes default to 1 for patient-wise aggregated FOV lists
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
        checkpoint_dir=checkpoint_dir,
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
            # Saves weights safely into the designated check-point file name
            trainer.save_checkpoint("best_neckcad_model.pt", val_metrics)

        if trainer.early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}. Training halted.")
            break


if __name__ == "__main__":
    main()
