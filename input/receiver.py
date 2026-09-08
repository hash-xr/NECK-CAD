import os
from PIL import Image, UnidentifiedImageError
from core.schema import ImageMetadata, ImageStatus
from traceability.logger import AuditLogger

class InputReceiver:
    def __init__(self, logger: AuditLogger):
        self.logger = logger

    def load_image(self, metadata: ImageMetadata) -> tuple[ImageMetadata, Image.Image | None]:
        """
        Reads image file from disk and populates raw metadata dimensions.
        Returns updated metadata and the raw PIL Image object.
        """
        if not os.path.exists(metadata.file_path):
            metadata.status = ImageStatus.REJECTED_TECHNICAL
            metadata.rejection_reason = f"File not found: {metadata.file_path}"
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="ERROR")
            return metadata, None

        try:
            pil_img = Image.open(metadata.file_path)
            pil_img.verify()  # Verify image integrity (catches corrupted files)
            
            # Re-open after verify() as PIL documentation recommends
            pil_img = Image.open(metadata.file_path)
            
            metadata.width, metadata.height = pil_img.size
            metadata.channels = len(pil_img.getbands())
            
            self.logger.log(f"[{metadata.image_id}] File loaded successfully ({metadata.width}x{metadata.height}, {metadata.channels} channels).")
            return metadata, pil_img

        except (UnidentifiedImageError, OSError) as e:
            metadata.status = ImageStatus.REJECTED_TECHNICAL
            metadata.rejection_reason = f"Corrupted or invalid image file: {str(e)}"
            self.logger.log(f"[{metadata.image_id}] {metadata.rejection_reason}", level="ERROR")
            return metadata, None