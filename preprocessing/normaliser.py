import torch
import torchvision.transforms as T
from typing import Callable
from PIL import Image
from config.settings import SystemConfig
from core.schema import ImageMetadata, ImageStatus
from traceability.logger import AuditLogger

class ImageNormaliser:
    def __init__(self, config: SystemConfig, logger: AuditLogger):
        self.config = config
        self.logger = logger
        
        # Standard PyTorch Preprocessing Pipeline
        self.transform_pipeline: Callable[[Image.Image], torch.Tensor] = T.Compose([
            T.Resize(self.config.TARGET_IMAGE_SIZE),
            T.ToTensor(),  # Scales PIL Image (0-255) to Tensor float (0.0 - 1.0)
            T.Normalize(
                mean=self.config.NORMALIZE_MEAN,
                std=self.config.NORMALIZE_STD
            )
        ])

    def normalize_stain(self, pil_img: Image.Image) -> Image.Image:
        """
        Optional Hook for Stain Normalization (e.g., Macenko / Vahadane).
        Can integrate torchstain or custom reference matrix here if required.
        """
        # Returns image unchanged if stain normalization is disabled or unavailable
        return pil_img

    def process(self, metadata: ImageMetadata, pil_img: Image.Image) -> ImageMetadata:
        """Resizes, standardizes, and converts PIL image to a PyTorch tensor payload."""
        if metadata.status != ImageStatus.VALID:
            return metadata

        try:
            # 1. Apply Stain Normalization Hook
            normalized_pil = self.normalize_stain(pil_img)
            
            # 2. PyTorch Tensor Transformations
            tensor: torch.Tensor = self.transform_pipeline(normalized_pil)
            
            # Add Batch Dimension [C, H, W] -> [1, C, H, W]
            metadata.processed_tensor = tensor.unsqueeze(0)
            self.logger.log(f"[{metadata.image_id}] Preprocessed tensor generated with shape {list(metadata.processed_tensor.shape)}.")
            
        except Exception as e:
            metadata.status = ImageStatus.REJECTED_QUALITY
            metadata.rejection_reason = f"Preprocessing failed: {str(e)}"
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="ERROR")

        return metadata