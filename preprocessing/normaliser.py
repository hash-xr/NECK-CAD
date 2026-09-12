import logging
import numpy as np
from typing import Callable, Optional
from PIL import Image

import torch
import torchvision.transforms as T

from config.settings import SystemConfig
from core.schema import ImageMetadata, ImageStatus
from traceability.logger import AuditLogger
from preprocessing.stain_normaliser import MacenkoStainNormaliser

logger = logging.getLogger("NECK-CAD.Normaliser")


class ImageNormaliser:
    """
    Standardises input slide FOVs through sequential Macenko stain 
    normalisation and PyTorch tensor feature scaling.
    """

    def __init__(self, config: SystemConfig, logger: Optional[AuditLogger] = None):
        self.config = config
        self.logger = logger
        
        # Instantiate Macenko Stain Engine
        self.stain_engine = MacenkoStainNormaliser(
            target_io=getattr(self.config, "STAIN_TARGET_IO", 240.0)
        )

        # Standard PyTorch Preprocessing Pipeline
        self.transform_pipeline: Callable[[Image.Image], torch.Tensor] = T.Compose([
            T.Resize(self.config.TARGET_IMAGE_SIZE),
            T.ToTensor(),  # Scales PIL Image (0-255) to Tensor float (0.0 - 1.0)
            T.Normalize(
                mean=self.config.NORMALISE_MEAN,
                std=self.config.NORMALISE_STD,
            ),
        ])

    def normalise_stain(self, pil_img: Image.Image) -> Image.Image:
        """
        Applies Macenko SVD stain normalisation on RGB image payload.
        """
        try:
            np_img = np.array(pil_img)
            stain_corrected_np = self.stain_engine.fit_transform(np_img)
            return Image.fromarray(stain_corrected_np)
        except Exception as e:
            if self.logger:
                self.logger.log(f"Stain normalisation bypass triggered: {str(e)}", level="WARNING")
            return pil_img

    def process(self, metadata: ImageMetadata, pil_img: Image.Image) -> ImageMetadata:
        """Resizes, standardises stain, and converts PIL image to a PyTorch tensor payload."""
        if metadata.status != ImageStatus.VALID:
            return metadata

        try:
            # 1. Stain Normalisation (Biological Color Alignment)
            normalised_pil = self.normalise_stain(pil_img)

            # 2. PyTorch Tensor Transformations (Feature Space Scaling)
            tensor: torch.Tensor = self.transform_pipeline(normalised_pil)

            # Add Batch Dimension [C, H, W] -> [1, C, H, W]
            metadata.processed_tensor = tensor.unsqueeze(0)

            if self.logger:
                self.logger.log(
                    f"[{metadata.image_id}] Preprocessed tensor generated with shape {list(metadata.processed_tensor.shape)}."
                )

        except Exception as e:
            metadata.status = ImageStatus.REJECTED_QUALITY
            metadata.rejection_reason = f"Preprocessing failed: {str(e)}"
            if self.logger:
                self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="ERROR")

        return metadata