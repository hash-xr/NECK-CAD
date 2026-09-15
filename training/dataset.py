import logging
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

logger = logging.getLogger("NECK-CAD.Dataset")


class SalivaryFNAFolderDataset(Dataset):
    """
    Dataset wrapper for processed slide feature tensors, morphology vectors, and targets.
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "embeddings": sample["embeddings"],       # [N, Dim]
            "morphology": sample["morphology"],       # [N, 9]
            "quality_scores": torch.tensor(sample["quality_scores"], dtype=torch.float32),
            "primary": torch.tensor(sample["primary"], dtype=torch.long),
            "milan": torch.tensor(sample["milan"], dtype=torch.long),
            "entity": torch.tensor(sample.get("entity", -1), dtype=torch.long),
            "patient_id": sample["patient_id"],
        }


def create_patient_stratified_splits(
    samples: List[Dict],
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits samples into Train, Val, and Test sets based strictly on patient IDs.
    """
    patient_ids = np.array([s["patient_id"] for s in samples])
    milan_labels = np.array([s["milan"] for s in samples])

    # First split: Train+Val vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    X_placeholder = np.zeros(len(samples))
    train_val_idx, test_idx = next(gss_test.split(X_placeholder, milan_labels, groups=patient_ids))

    train_val_samples = [samples[i] for i in train_val_idx]
    test_samples = [samples[i] for i in test_idx]

    # Second split: Train vs Val
    adj_val_size = val_size / (1.0 - test_size)
    tv_patient_ids = np.array([s["patient_id"] for s in train_val_samples])
    tv_milan_labels = np.array([s["milan"] for s in train_val_samples])

    gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val_size, random_state=random_state)

    X_tv_placeholder = np.zeros(len(train_val_samples))
    train_idx, val_idx = next(gss_val.split(X_tv_placeholder, tv_milan_labels, groups=tv_patient_ids))

    train_samples = [train_val_samples[i] for i in train_idx]
    val_samples = [train_val_samples[i] for i in val_idx]

    logger.info(
        f"Data split complete: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)} cases."
    )

    return (
        SalivaryFNAFolderDataset(train_samples),
        SalivaryFNAFolderDataset(val_samples),
        SalivaryFNAFolderDataset(test_samples),
    )