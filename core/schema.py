from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class ImageStatus(Enum):
    PENDING = "PENDING"
    VALID = "VALID"
    REJECTED_TECHNICAL = "REJECTED_TECHNICAL"
    REJECTED_QUALITY = "REJECTED_QUALITY"

@dataclass
class ImageMetadata:
    image_id: str
    file_path: str
    filename: str
    status: ImageStatus = ImageStatus.PENDING
    rejection_reason: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 0
    quality_score: float = 0.0
    processed_tensor: Optional[Any] = None

@dataclass
class DiagnosticResult:
    primary_category: str = "Unknown"
    milan_category: str = "Unknown"
    specific_diagnosis: str = "Unknown"
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    differential_diagnosis: List[Tuple[str, float]] = field(default_factory=list)
    is_uncertain: bool = False
    is_ood: bool = False

@dataclass
class AnalysisSession:
    session_id: str
    images: List[ImageMetadata] = field(default_factory=list)
    aggregated_result: Optional[DiagnosticResult] = None
    explanations: Dict[str, np.ndarray] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)