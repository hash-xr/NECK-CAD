from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SystemConfig:
    # Supported File Formats & Dimensions
    SUPPORTED_FORMATS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    MIN_IMAGE_DIMENSIONS: Tuple[int, int] = (224, 224)
    REQUIRED_CHANNELS: int = 3
    
    # Image Quality Thresholds (Laplacian Variance for Sharpness)
    SHARPNESS_THRESHOLD: float = 100.0
    
    # Preprocessing
    TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)
    NORMALISE_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    NORMALISE_STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Classification & Uncertainty
    CONFIDENCE_THRESHOLD: float = 0.60
    OOD_ENERGY_THRESHOLD: float = -5.0

config = SystemConfig()