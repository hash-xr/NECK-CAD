import logging
from dataclasses import dataclass
import numpy as np
import cv2
from typing import Optional

from traceability.logger import AuditLogger

logger = logging.getLogger("NECK-CAD.Artifacts")


@dataclass
class ArtifactAssessment:
    """Quantitative artifact flags and severity scores for an FNA slide FOV."""

    rbc_coverage_ratio: float  # Fraction of FOV obscured by red blood cells
    air_drying_index: float    # Measure of nuclear edge blur due to slow drying
    debris_coverage_ratio: float  # Fraction of background covered by gel/protein debris
    is_usable: bool            # True if FOV passes artifact quality gates
    flag_reasons: list         # List of triggered artifact warnings


class CytologyArtifactDetector:
    """Detects blood, air-drying, and debris artifacts in FNA cytopathology smears."""

    def __init__(
        self,
        logger: Optional[AuditLogger] = None,
        max_rbc_ratio: float = 0.40,
        max_debris_ratio: float = 0.35,
        min_edge_sharpness: float = 12.0,
    ):
        self.logger = logger
        self.max_rbc_ratio = max_rbc_ratio
        self.max_debris_ratio = max_debris_ratio
        self.min_edge_sharpness = min_edge_sharpness

    def assess_fov(self, img_rgb: np.ndarray, image_id: Optional[str] = None) -> ArtifactAssessment:
        """
        Evaluates input FOV image for cytological artifacts.

        Args:
            img_rgb (np.ndarray): HxWx3 uint8 RGB slide image.

        Returns:
            ArtifactAssessment: Structured artifact metrics and usability flags.
        """
        flags = []
        prefix = f"[{image_id}] " if image_id else ""

        # 1. RBC / Blood Contamination Detection (HSV Red Channel Masking)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Red hue spans lower and upper spectrum
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        rbc_mask = cv2.bitwise_or(mask1, mask2)
        
        rbc_ratio = float(np.sum(rbc_mask > 0) / (img_rgb.shape[0] * img_rgb.shape[1]))
        if rbc_ratio > self.max_rbc_ratio:
            flags.append(f"Excessive RBC/Blood contamination ({rbc_ratio:.1%} of FOV)")

        # 2. Air-Drying Artifact Index (Loss of high-frequency nuclear edges)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        air_drying_index = float(np.mean(edge_magnitude))

        if air_drying_index < self.min_edge_sharpness:
            flags.append(f"Potential air-drying artifact detected (Sharpness: {air_drying_index:.1f})")

        # 3. Proteinaceous Background Debris & Gel Masking
        # High saturation + mid-intensity background regions
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        debris_mask = (s_channel < 40) & (v_channel > 100) & (v_channel < 210)
        debris_ratio = float(np.sum(debris_mask) / (img_rgb.shape[0] * img_rgb.shape[1]))

        if debris_ratio > self.max_debris_ratio:
            flags.append(f"High background debris/gel occlusion ({debris_ratio:.1%} of FOV)")

        is_usable = len(flags) == 0

        if self.logger:
            if is_usable:
                self.logger.log(f"{prefix}Artifact assessment passed (RBC: {rbc_ratio:.1%}, Debris: {debris_ratio:.1%}).")
            else:
                self.logger.log(f"{prefix}Artifact flags triggered: {', '.join(flags)}", level="WARNING")

        return ArtifactAssessment(
            rbc_coverage_ratio=rbc_ratio,
            air_drying_index=air_drying_index,
            debris_coverage_ratio=debris_ratio,
            is_usable=is_usable,
            flag_reasons=flags,
        )