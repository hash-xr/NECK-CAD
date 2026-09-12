import cv2
import numpy as np
from PIL import Image
from config.settings import SystemConfig
from core.schema import ImageMetadata, ImageStatus
from traceability.logger import AuditLogger

class QualityEvaluator:
    def __init__(self, config: SystemConfig, logger: AuditLogger):
        self.config = config
        self.logger = logger

    def calculate_sharpness(self, cv_image: np.ndarray) -> float:
        """Computes variance of Laplacian as a proxy for image sharpness/focus."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def check_exposure(self, cv_image: np.ndarray) -> tuple[bool, str]:
        """Checks if image is severely overexposed or underexposed (blank/dark slides)."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)
        
        if mean_val < 15.0:
            return False, f"Image is underexposed or blank (mean intensity: {mean_val:.1f})"
        if mean_val > 245.0:
            return False, f"Image is overexposed or background-only (mean intensity: {mean_val:.1f})"
            
        return True, "OK"

    def evaluate_quality(self, metadata: ImageMetadata, pil_img: Image.Image) -> ImageMetadata:
        """Gates images based on focus quality and exposure analytical suitability."""
        if metadata.status != ImageStatus.VALID:
            return metadata

        # Convert PIL Image to OpenCV RGB NumPy Array
        cv_img = np.array(pil_img.convert("RGB"))

        # 1. Exposure & Analytical Suitability Check
        is_valid_exposure, exposure_reason = self.check_exposure(cv_img)
        if not is_valid_exposure:
            metadata.status = ImageStatus.REJECTED_QUALITY
            metadata.rejection_reason = exposure_reason
            self.logger.log(f"[{metadata.image_id}] Quality Rejection: {exposure_reason}", level="WARNING")
            return metadata

        # 2. Sharpness / Focus Assessment
        sharpness_score = self.calculate_sharpness(cv_img)
        metadata.quality_score = sharpness_score

        if sharpness_score < self.config.SHARPNESS_THRESHOLD:
            metadata.status = ImageStatus.REJECTED_QUALITY
            metadata.rejection_reason = f"Image out of focus. Sharpness ({sharpness_score:.2f}) below threshold ({self.config.SHARPNESS_THRESHOLD:.2f})."
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="WARNING")
            return metadata

        self.logger.log(f"[{metadata.image_id}] Passed quality gating. Sharpness score: {sharpness_score:.2f}.")
        return metadata