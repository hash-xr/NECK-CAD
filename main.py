import os
import uuid
from PIL import Image
from config.settings import config
from core.schema import AnalysisSession, ImageMetadata, ImageStatus
from traceability.logger import AuditLogger
from input.receiver import InputReceiver
from validation.validator import TechnicalValidator
from quality.evaluator import QualityEvaluator
from preprocessing.normaliser import ImageNormaliser

class NeckCADController:
    def __init__(self):
        self.config = config
        self.logger = AuditLogger()
        self.receiver = InputReceiver(self.logger)
        self.validator = TechnicalValidator(self.config, self.logger)
        self.evaluator = QualityEvaluator(self.config, self.logger)
        self.normaliser = ImageNormaliser(self.config, self.logger)
        self.logger.log("NECK-CAD Controller Initialized with Quality & Preprocessing engines.")

    def create_session(self, file_paths: list[str]) -> AnalysisSession:
        session_id = str(uuid.uuid4())[:8]
        session = AnalysisSession(session_id=session_id)
        
        for index, path in enumerate(file_paths):
            image_id = f"IMG_{session_id}_{index+1:02d}"
            metadata = ImageMetadata(
                image_id=image_id,
                file_path=path,
                filename=os.path.basename(path)
            )
            session.images.append(metadata)
            
        self.logger.log(f"Session {session_id} created with {len(file_paths)} image(s).")
        return session

    def process_session(self, session: AnalysisSession) -> AnalysisSession:
        """Executes ingestion, technical validation, quality gating, and preprocessing."""
        for metadata in session.images:
            # 1. Ingestion
            metadata, raw_img = self.receiver.load_image(metadata)
            if raw_img is None or metadata.status != ImageStatus.PENDING:
                continue

            # 2. Technical Validation
            metadata = self.validator.validate(metadata, raw_img)
            if metadata.status != ImageStatus.VALID:
                continue

            # 3. Quality Control (Focus & Exposure Gating)
            metadata = self.evaluator.evaluate(metadata, raw_img)
            if metadata.status != ImageStatus.VALID:
                continue

            # 4. Preprocessing & Tensor Conversion
            metadata = self.normaliser.process(metadata, raw_img)

        valid_count = sum(1 for img in session.images if img.status == ImageStatus.VALID)
        self.logger.log(f"Session {session.session_id}: {valid_count}/{len(session.images)} images preprocessed and ready for diagnostic engine.")
        return session

if __name__ == "__main__":
    controller = NeckCADController()
    print("Phase 3 complete. Quality Gating and Preprocessing integrated.")