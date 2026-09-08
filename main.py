import uuid
from PIL import Image
from config.settings import config
from core.schema import AnalysisSession, ImageMetadata, ImageStatus
from traceability.logger import AuditLogger
from input.receiver import InputReceiver
from validation.validator import TechnicalValidator

class NeckCADController:
    def __init__(self):
        self.config = config
        self.logger = AuditLogger()
        self.receiver = InputReceiver(self.logger)
        self.validator = TechnicalValidator(self.config, self.logger)
        self.logger.log("NECK-CAD Application Controller initialized.")

    def create_session(self, file_paths: list[str]) -> AnalysisSession:
        session_id = str(uuid.uuid4())[:8]
        session = AnalysisSession(session_id=session_id)
        
        for index, path in enumerate(file_paths):
            image_id = f"IMG_{session_id}_{index+1:02d}"
            filename = os.path.basename(path)
            metadata = ImageMetadata(
                image_id=image_id,
                file_path=path,
                filename=filename
            )
            session.images.append(metadata)
            
        self.logger.log(f"Session {session_id} created with {len(file_paths)} image(s).")
        return session

    def process_technical_stage(self, session: AnalysisSession) -> tuple[AnalysisSession, dict[str, Image.Image]]:
        """
        Runs ingestion and technical validation across all images in session.
        Returns updated session and a dictionary of loaded PIL images for valid files.
        """
        loaded_images: dict[str, Image.Image] = {}

        for metadata in session.images:
            # Step 1: Ingestion
            metadata, raw_img = self.receiver.load_image(metadata)
            
            # Step 2: Technical Validation
            metadata = self.validator.validate(metadata, raw_img)
            
            if metadata.status == ImageStatus.VALID and raw_img is not None:
                loaded_images[metadata.image_id] = raw_img

        valid_count = sum(1 for img in session.images if img.status == ImageStatus.VALID)
        self.logger.log(f"Session {session.session_id}: {valid_count}/{len(session.images)} images passed technical validation.")
        return session, loaded_images

if __name__ == "__main__":
    import os
    controller = NeckCADController()
    print("Phase 2 components successfully integrated.")