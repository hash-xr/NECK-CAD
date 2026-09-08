import os
from PIL import Image
from config.settings import SystemConfig
from core.schema import ImageMetadata, ImageStatus
from traceability.logger import AuditLogger

class TechnicalValidator:
    def __init__(self, config: SystemConfig, logger: AuditLogger):
        self.config = config
        self.logger = logger

    def validate(self, metadata: ImageMetadata, raw_img: Image.Image | None) -> ImageMetadata:
        """
        Executes technical validation checks on standard image properties.
        """
        if metadata.status == ImageStatus.REJECTED_TECHNICAL or raw_img is None:
            return metadata

        # 1. File Extension Verification
        ext = os.path.splitext(metadata.filename)[1].lower()
        if ext not in self.config.SUPPORTED_FORMATS:
            metadata.status = ImageStatus.REJECTED_TECHNICAL
            metadata.rejection_reason = f"Unsupported file extension '{ext}'. Expected one of {self.config.SUPPORTED_FORMATS}"
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="WARNING")
            return metadata

        # 2. Minimum Spatial Dimensions Check
        min_w, min_h = self.config.MIN_IMAGE_DIMENSIONS
        if metadata.width < min_w or metadata.height < min_h:
            metadata.status = ImageStatus.REJECTED_TECHNICAL
            metadata.rejection_reason = f"Image dimensions ({metadata.width}x{metadata.height}) fall below minimum required ({min_w}x{min_h})."
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="WARNING")
            return metadata

        # 3. Channel Count Verification (RGB Requirement)
        if metadata.channels != self.config.REQUIRED_CHANNELS:
            metadata.status = ImageStatus.REJECTED_TECHNICAL
            metadata.rejection_reason = f"Invalid channel count ({metadata.channels}). Required: {self.config.REQUIRED_CHANNELS} (RGB)."
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="WARNING")
            return metadata

        # Passed all technical validation checks
        metadata.status = ImageStatus.VALID
        self.logger.log(f"[{metadata.image_id}] Passed technical validation.")
        return metadata