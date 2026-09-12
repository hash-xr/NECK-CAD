from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class ImageStatus(Enum):
    PENDING = "PENDING"
    VALID = "VALID"
    REJECTED_TECHNICAL = "REJECTED_TECHNICAL"
    REJECTED_QUALITY = "REJECTED_QUALITY"

    # NEW: main.py uses these two generic statuses directly (INVALID for
    # failed technical validation, REJECTED for failed quality control).
    # Kept alongside the original two so any code still checking for
    # REJECTED_TECHNICAL / REJECTED_QUALITY keeps working unchanged.
    INVALID = "INVALID"
    REJECTED = "REJECTED"


class MilanCategory(Enum):
    """The Milan System for Reporting Salivary Gland Cytopathology categories."""
    I_NON_DIAGNOSTIC = "I: Non-Diagnostic"
    II_NON_NEOPLASTIC = "II: Non-Neoplastic"
    III_AUS = "III: Atypia of Undetermined Significance"
    IVA_NEOPLASM_BENIGN = "IVA: Neoplasm - Benign"
    IVB_SUMP = "IVB: Salivary Gland Neoplasm of Uncertain Malignant Potential"
    V_SUSPICIOUS = "V: Suspicious for Malignancy"
    VI_MALIGNANT = "VI: Malignant"


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
class AggregatedResult:
    """Session-level diagnostic result after multi-FOV fusion + classification."""
    primary_category: str = "Unknown"
    milan_category: str = "Unknown"
    specific_diagnosis: str = "Unknown"
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    differential_diagnosis: List[Tuple[str, float]] = field(default_factory=list)
    is_uncertain: bool = False
    is_ood: bool = False


# Backward-compat alias: old code importing DiagnosticResult keeps working.
DiagnosticResult = AggregatedResult


@dataclass
class DiagnosticSession:
    session_id: str
    images: List[ImageMetadata] = field(default_factory=list)
    aggregated_result: Optional[AggregatedResult] = None
    explanations: Dict[str, np.ndarray] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


# Backward-compat alias: old code importing AnalysisSession keeps working.
AnalysisSession = DiagnosticSession