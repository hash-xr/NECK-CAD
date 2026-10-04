# evaluate_model.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from config.settings import SystemConfig
from main import NeckCADController
from training.dataset import SalivaryFNAFolderDataset, create_patient_stratified_splits
from training.trainer import MILAN_CATEGORY_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NECK-CAD.Evaluation")

PRIMARY_CATEGORY_NAMES = ["Non-Diagnostic", "Non-Neoplastic", "Neoplastic"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str,
    output_path: str,
):
    """Generates and saves a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to '{output_path}'.")


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    title: str,
    output_path: str,
):
    """Computes and plots One-vs-Rest (OvR) ROC curves and AUC for each class."""
    n_classes = len(class_names)
    # FIX: Explicitly convert to NumPy array to clear Pylance spmatrix typing conflicts
    y_true_bin = np.asarray(label_binarize(y_true, classes=list(range(n_classes))))
    y_probs_arr = np.asarray(y_probs)

    plt.figure(figsize=(10, 8))
    roc_auc = {}

    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_arr[:, i])
            roc_auc[i] = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
            )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    # FIX: Replaced list brackets with float positional parameters
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves plot to '{output_path}'.")


@torch.no_grad()
def evaluate_test_set(
    controller: NeckCADController,
    test_loader: DataLoader,
    output_dir: str = "./evaluation_results",
):
    """Runs model evaluation on the test dataset split and generates performance artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    controller.fusion_engine.eval()
    controller.classifier.eval()

    milan_targets, milan_preds, milan_probs_list = [], [], []
    primary_targets, primary_preds, primary_probs_list = [], [], []

    logger.info("Starting test set inference...")

    for batch in test_loader:
        embeddings = batch["embeddings"].squeeze(0).to(controller.device)  # [N, Dim]
        morphology = batch["morphology"].squeeze(0).to(controller.device)  # [N, 9]
        quality_scores = batch["quality_scores"].squeeze(0).tolist()

        case_morphology = torch.mean(morphology, dim=0, keepdim=True)
        case_embedding, _ = controller.fusion_engine(embeddings, quality_scores=quality_scores)

        outputs = controller.classifier(case_embedding, case_morphology)

        # Collect Milan category outputs
        m_probs = outputs["milan_probs"].squeeze(0).cpu().numpy()
        m_pred = int(np.argmax(m_probs))
        milan_probs_list.append(m_probs)
        milan_preds.append(m_pred)
        milan_targets.append(int(batch["milan"]))

        # Collect Primary category outputs
        p_probs = outputs["primary_probs"].squeeze(0).cpu().numpy()
        p_pred = int(np.argmax(p_probs))
        primary_probs_list.append(p_probs)
        primary_preds.append(p_pred)
        primary_targets.append(int(batch["primary"]))

    y_milan_true = np.array(milan_targets)
    y_milan_pred = np.array(milan_preds)
    y_milan_prob = np.array(milan_probs_list)

    y_primary_true = np.array(primary_targets)
    y_primary_pred = np.array(primary_preds)
    y_primary_prob = np.array(primary_probs_list)

    # 1. Classification Metrics Summary
    print("\n=================== MILAN RISK CATEGORY METRICS ===================")
    print(classification_report(y_milan_true, y_milan_pred, labels=list(range(len(MILAN_CATEGORY_NAMES))), target_names=MILAN_CATEGORY_NAMES, zero_division=0))
    print(f"Overall Milan Accuracy: {accuracy_score(y_milan_true, y_milan_pred):.4f}")
    print(f"Macro F1-Score: {f1_score(y_milan_true, y_milan_pred, average='macro', zero_division=0):.4f}")

    print("\n=================== PRIMARY CATEGORY METRICS ===================")
    print(classification_report(y_primary_true, y_primary_pred, labels=list(range(len(PRIMARY_CATEGORY_NAMES))), target_names=PRIMARY_CATEGORY_NAMES, zero_division=0))

    # 2. Visual Artifact Generation
    plot_confusion_matrix(
        y_milan_true,
        y_milan_pred,
        MILAN_CATEGORY_NAMES,
        "Normalized Confusion Matrix - Milan System",
        os.path.join(output_dir, "milan_confusion_matrix.png"),
    )

    plot_multiclass_roc_curves(
        y_milan_true,
        y_milan_prob,
        MILAN_CATEGORY_NAMES,
        "One-vs-Rest ROC Curves - Milan System",
        os.path.join(output_dir, "milan_roc_curves.png"),
    )

    plot_confusion_matrix(
        y_primary_true,
        y_primary_pred,
        PRIMARY_CATEGORY_NAMES,
        "Normalized Confusion Matrix - Primary Category",
        os.path.join(output_dir, "primary_confusion_matrix.png"),
    )

    logger.info(f"All evaluation artifacts saved to '{output_dir}'.")


if __name__ == "__main__":
    config = SystemConfig()
    controller = NeckCADController(config=config)

    # FIX: Replicate training stratification exactly to avoid data leakage
    processed_data_path = "data/processed_features.pt"
    
    if os.path.exists(processed_data_path):
        logger.info(f"Loading cached features to construct validation split from: '{processed_data_path}'")
        all_samples = torch.load(processed_data_path)
        
        # Pull splitting seed from system configs
        random_seed = getattr(config, "RANDOM_STATE", 42)
        
        # Split using identical parameters as run_training.py
        _, _, test_dataset = create_patient_stratified_splits(
            all_samples, val_size=0.2, test_size=0.1, random_state=random_seed
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        evaluate_test_set(controller, test_loader)
    else:
        logger.warning(
            f"Feature matrix map file not found at '{processed_data_path}'! "
            f"Run 'extract_features.py' before executing evaluations."
        )